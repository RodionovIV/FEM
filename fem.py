import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class Data:

    def __init__(self):
        self.__dim = None
        self.__poisson = None
        self.__youngModulus = None
        self.__n_nodes_ = None
        self.__nodes = None
        self.__n_elements = None
        self.__elements = None
        self.__n_constraints = None
        self.__constraints = None
        self.__n_loads = None
        self.__loads = None
        self.__D_matrix = None

    @property
    def nodes(self):
        return self.__n_nodes, self.__nodes

    @property
    def elements(self):
        return self.__n_elements, self.__elements

    @property
    def constraints(self):
        return self.__n_constraints, self.__constraints

    @property
    def loads(self):
        return self.__n_loads, self.__loads

    @property
    def D(self):
        p,y = self.__poisson, self.__youngModulus
        self.__D_matrix = np.array([[1.,p,0.],[p,1.,0.],[0.,0.,(1 - p)/2]]) * (y/(1 - p**2))
        return self.__D_matrix

    @property
    def dim(self):
        return self.__dim

    @nodes.setter
    def nodes(self, filename = None):
        if filename is None:
            raise FileNotFoundError('Filename not specified {nodes}')
        with open(filename,'r') as nodes_file:
            self.__n_nodes = int(nodes_file.readline())
            self.__nodes = np.array([list(map(float, line.split())) for line in nodes_file])

    @elements.setter
    def elements(self, filename = None):
        if filename is None:
            raise FileNotFoundError('Filename not specified {elements}')
        with open(filename,'r') as elements_file:
            self.__n_elements = int(elements_file.readline())
            self.__elements = np.array([list(map(int, line.split())) for line in elements_file])


    @constraints.setter
    def constraints(self, filename = None):
        if filename is None:
            raise FileNotFoundError('Filename not specified {constraints}')
        with open(filename,'r') as constraints_file:
            self.__n_constraints = int(constraints_file.readline())
            self.__constraints = np.array([list(map(int, line.split())) for line in constraints_file])

    @loads.setter
    def loads(self, filename = None):
        if filename is None:
            raise FileNotFoundError('Filename not specified {loads}')
        if filename == 'stress.txt':
            elements, nodes = self.__elements, self.__nodes
            dot = lambda v1, v2: v1[0]*v2[0] + v1[1]*v2[1]
            l = lambda v: dot(v,v)**0.5
            stress = open(filename,'r')
            stress_count = int(stress.readline())
            loads = open('loads.txt','w')
            loads.write(f'{stress_count}')
            completed = dict()
            for line in stress:
                e1,e2,element, nx, ny, P = list(map(float, line.split()))
                e1,e2,element = map(int,[e1,e2,element])
                e3 = elements[element][2]
                nx, ny = nx/l((nx,ny)), ny/l((nx,ny))
                x1, y1 = nodes[e1]
                x2, y2 = nodes[e2]
                x3, y3 = nodes[e3]
                check_x, check_y = x3 - x1, y3 - y1
                lx, ly = x2 - x1, y2 - y1

                cos = dot((nx,ny), (check_x,check_y))

                if cos > 0:
                    nx, ny = (-1)*nx, (-1)*ny
                L = l((lx, ly))


                if e1 in completed.keys():
                    completed[e1] += np.array([nx,ny])
                else:
                    completed[e1] = np.array([nx,ny])

                if e2 in completed.keys():
                    completed[e2] += np.array([nx,ny])
                else:
                    completed[e2] = np.array([nx,ny])
            for key, value in zip(completed.keys(), completed.values()):
                     loads.write(f'\n{key} {(-0.5)*value[0]*P*L} {(-0.5)*value[1]*P*L}')
            stress.close()
            loads.close()
            filename = 'loads.txt'

        with open(filename,'r') as loads_file:
            dim = self.__dim
            self.__n_loads = loads_count = int(loads_file.readline())
            self.__loads = np.zeros(dim * self.__n_nodes)
            for line in loads_file:
                node, load_x, load_y = [float(x) for x in line.split()]
                self.__loads[dim*int(node) + 0], self.__loads[dim*int(node) + 1] = load_x, load_y


    def make(self, filename=None):
        if filename == None:
            raise FileExistsError('Filename not specified! {config}')
        with open(filename,'r') as config_file:
            files = [line.split('\n')[0] for line in config_file]
            self.nodes, self.elements, self.constraints, self.loads = files


    def set_params(self, dim = None ,poisson = None, youngModulus = None ,set_default = False):
        if (dim is None) or (poisson is None) or (youngModulus is None) or (set_default == True):
            self.__dim = 2
            self.__poisson = 0.25
            self.__youngModulus = 2e+7
        else:
            self.__dim = dim
            self.__poisson = poisson
            self.__youngModulus = youngModulus

    def get_params(self):
        return {'dim':self.__dim, 'poisson':self.__poisson, 'youngModulus':self.__youngModulus}

    def info(self):
        print(f'dim: {self.__dim}\npoisson: {self.__poisson}\nyoungModulus: {self.__youngModulus}')


    def get_nodes(self):
        return self.__n_nodes, self.__nodes
    def get_elements(self):
        return self.__n_elements, self.__elements
    def get_constraints(self):
        return self.__n_constraints, self.__constraints
    def get_loads(self):
        return self.__n_loads, self.__loads
    def get_D(self):
        return self.__D_matrix

#Класс, отвечающий за решение задачи
class Solver:

    def __init__(self):
        self.__disp = None
        self.__eps = None
        self.__sigma = None
        self.__sigma_mises = None


#Основная вычислительная функция, здесь происходит расчет и преобразование глобальной матрицы жесткости, применение закреплений и вычисление перемещений
    def process(self, data):
        #Выгружаем данные
        n_nodes, nodes = data.nodes
        n_elements, elements = data.elements
        n_constraints, constraints = data.constraints
        n_loads, loads = data.loads
        D = data.D
        dim = data.dim

        #Собираем глобальную матрицу жесткости
        #CalculateStiffnessMatrix{
        nodes_x, nodes_y = nodes.T
        K_rows, K_cols, K_values = [], [], []

        for element in elements:

            x = np.array([nodes_x[element[0]], nodes_x[element[1]], nodes_x[element[2]]])
            y = np.array([nodes_y[element[0]], nodes_y[element[1]], nodes_y[element[2]]])

            C = np.matrix([np.ones(3),x,y])
            C = C.T
            IC = C.I

            B = np.zeros((3,6))

            for i in range(3):
                B[0, 2 * i + 0] = IC[1,i]
                B[1, 2 * i + 1] = IC[2,i]
                B[2, 2 * i + 0] = IC[2,i]
                B[2, 2 * i + 1] = IC[1,i]

            loc_K = np.matmul(np.matmul(B.T, D), B) * (np.linalg.det(C) / 2)

            for i in range(3):
                for j in range(3):
                        K_rows.extend([2 * element[i] + 0, 2 * element[i] + 0, 2 * element[i] + 1, 2 * element[i] +1])
                        K_cols.extend([2 * element[j] + 0, 2 * element[j] + 1, 2 * element[j] + 0, 2 * element[j] +1])
                        K_values.extend([loc_K[2*i + 0, 2*j + 0],loc_K[2*i + 0, 2*j + 1],loc_K[2*i + 1, 2*j + 0],loc_K[2*i + 1, 2*j + 1]])

        #CalculateStiffnessMatrix}

        #Преобразовываем матрицу K, суммируем значения по индексам
        K_rows = np.array(K_rows)
        K_cols = np.array(K_cols)
        K_values = np.array(K_values)
        coord=np.vstack((K_rows, K_cols))
        u, indices = np.unique(coord, return_inverse=True, axis=1)
        K_values=np.bincount(indices, weights=K_values)
        K_rows, K_cols=np.vsplit(u, 2)
        N = dim * n_nodes
        K = np.array([K_rows[0], K_cols[0], K_values]).T

        #Применяем закрепления (ApplyConstraints){
        ind2Constraint = [2 * i[0] + 0 for i in constraints if i[1] == 1 or i[1] == 3] + [2 * i[0] + 1 for i in constraints if i[1] == 2 or i[1] == 3]
        #diag_K = np.array([line for line in K if line[0] == line[1]])
        diag_K = [line for line in K if line[0] == line[1]]
        other_K = np.array([line for line in K if line[0] != line[1]])
        #nn_K = np.array([[line[0], line[1], 1] if line[0] in ind2Constraint else line for line in diag_K] + [[line[0],line[1],0] if line[0] in ind2Constraint or line[1] in ind2Constraint else line for line in other_K]).T
        nn_K = np.array(diag_K + [[line[0],line[1],0] if line[0] in ind2Constraint or line[1] in ind2Constraint else line for line in other_K]).T
        rows,cols,values = nn_K
        A =  csr_matrix((values, (rows, cols)), shape = (N,N))
        #ApplyConstraints}

        #Считаем перемещения
        disp = spsolve(A, loads).reshape((-1,dim))

        #Считаем деформации и напряжения

        sigma = []
        sigma_mises = []
        eps = []
        mises = lambda sigma: sigma[0][0]**2 - sigma[0][0]*sigma[0][1] + sigma[0][1]**2 + 3*sigma[0][2]**2

        for element in elements:

            x = np.array([nodes_x[element[0]], nodes_x[element[1]], nodes_x[element[2]]])
            y = np.array([nodes_y[element[0]], nodes_y[element[1]], nodes_y[element[2]]])

            C = np.matrix([np.ones(3),x,y])
            C = C.T
            IC = C.I

            B = np.zeros((3,6))

            for i in range(3):
                B[0, 2 * i + 0] = IC[1,i]
                B[1, 2 * i + 1] = IC[2,i]
                B[2, 2 * i + 0] = IC[2,i]
                B[2, 2 * i + 1] = IC[1,i]

            delta = np.vstack([disp[element[0]], disp[element[1]], disp[element[2]]]).reshape((-1,1))
            eps.extend([np.matmul(B,delta)])
            sigma.extend([np.matmul(D,eps[-1]).reshape((1,-1))])
            sigma_mises.extend([mises(sigma[-1])])

        self.__disp, self.__eps, self.__sigma, self.__sigma_mises = disp, eps, sigma, sigma_mises

        return sigma

    def visual_stress(self, data, mises = False):

        n_elements, elements = data.elements
        n_nodes, nodes = data.nodes
        sigma = self.__sigma
        sigma_xx = [j[0][0] for j in sigma]
        sigma_yy = [j[0][1] for j in sigma]
        sigma_xy = [j[0][2] for j in sigma]
        nodes_x, nodes_y = nodes.T

        size = (20, 10)

        plt.figure(figsize = size)
        plt.title('Sigma X', fontsize = 15)
        plt.tripcolor(nodes_x, nodes_y, elements, sigma_xx)
        plt.colorbar()
        plt.savefig('1.png')
        plt.close()

        plt.figure(figsize = size)
        plt.title('Sigma Y', fontsize = 15)
        plt.tripcolor(nodes_x, nodes_y, elements, sigma_yy)
        plt.colorbar()
        plt.savefig('2.png')
        plt.close()

        plt.figure(figsize = size)
        plt.title('Sigma XY', fontsize = 15)
        plt.tripcolor(nodes_x, nodes_y, elements, sigma_xy)
        plt.colorbar()
        plt.savefig('3.png')
        plt.close()

        if mises == True:
            sigma_mises = self.__sigma_mises
            plt.figure(figsize = size)
            plt.title('Sigma Mises', fontsize = 15)
            plt.tripcolor(nodes_x, nodes_y, elements, sigma_mises)
            plt.colorbar()
            plt.savefig('4.png')
            plt.close()
