from fem import Data
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy.linalg as LA
import pandas as pd





def in_triangle(x,y,element, nodes):
    x0, y0 = x,y
    x1,y1 = nodes[element[0]]
    x2,y2 = nodes[element[1]]
    x3,y3 = nodes[element[2]]

    r1 = (x1 - x0) * (y2 - y1) - (x2 - x1) * (y1 - y0)
    r2 = (x2 - x0) * (y3 - y2) - (x3 - x2) * (y2 - y0)
    r3 = (x3 - x0) * (y1 - y3) - (x1 - x3) * (y3 - y0)

    if (r1 <= 0 and r2 <= 0 and r3 <= 0) or (r1 >= 0 and r2 >= 0 and r3 >= 0):
        return True
    else:
        return False


def interpolateSigma(x, y, sigma, axis, nodes, elements):
    s = None
    for key, val in enumerate(elements):
            if in_triangle(x,y,val,nodes):
                s = sigma[key][0][axis]
                break
    return s
def interpolateSigmaMises(x, y, sigma_mises, nodes, elements):
    s = None
    for key, val in enumerate(elements):
            if in_triangle(x,y,val,nodes):
                s = sigma[key]
                break
    return s


def methodAR(res_value, nodes, elements):

    C = np.zeros(shape=(len(nodes), len(nodes)))
    R = np.zeros(len(nodes))

    for key, val in enumerate(elements):
        x1, y1 = nodes[val[0]]
        x2, y2 = nodes[val[1]]
        x3, y3 = nodes[val[2]]

        dlt = abs((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)) / 2

        R[[val[0], val[1], val[2]]] += res_value[key] * np.abs(dlt) * 3

        for n1, n2 in product([val[0], val[1], val[2]],
                              [val[0], val[1], val[2]]):
            C[n1, n2] += dlt

    res = np.linalg.solve(C, R)

    return res

def interpolation(x, y, nodes_value, nodes, elements):
    ins = None
    for i,element in enumerate(elements):
        if in_triangle(x, y, element, nodes):
            ins = i
            break
    if ins == None:
        print('Check interval')

    F = [nodes_value[elements[ins][i]] for i in range(3)]
    X = [nodes[elements[ins][i]][0] for i in range(3)]
    Y = [nodes[elements[ins][i]][1] for i in range(3)]


    b = [Y[i%3] - Y[(i+1)%3] for i in range(1, 4)]
    c = [-X[i%3] + X[(i+1)%3] for i in range(1, 4)]
    a = [X[i%3] * Y[(i+1)%3] - X[(i+1)%3] * Y[i%3] for i in range(1, 4)]
    A = np.linalg.det(np.array([
                    [1, X[0], Y[0]],
                    [1, X[1], Y[1]],
                    [1, X[2], Y[2]]
                        ]))
    res = 0
    for i in range(3):
        res += (a[i] + b[i] * x + c[i] * y) / A * F[i]
    return res


def charts_results(df, fixed, mesh_count, name = None):
    if name == None:
        name = ''
    N = 40
    fixed_x, fixed_y = fixed
    for fixed_axes,axes,n in zip(['X','Y'], ['Y','X'], range(2)):
        for component in ['XX','YY','XY']:
            plt.figure(figsize = (20,10))
            for i in range(1,mesh_count):
                plt.plot(df[axes], df[f'mesh{i}_{component}_{axes.lower()}'])
                plt.ylabel(f'Sigma {component}')
                plt.legend([f'mesh-{i}' for i in range(1,mesh_count)])
            plt.title(f'{fixed_axes} = {fixed[n]}')
            plt.savefig(f'{name}_{component}_{fixed_axes.lower()}')
            plt.close()

def print_results(df, mesh_count):
    for axes in ['X','Y']:
        for component in ['XX','YY','XY']:
            print(f'{axes}:{component}')
            for i in range(2,mesh_count):
                cur = df[f'mesh{i}_{component}_{axes.lower()}']
                prev = df[f'mesh{i-1}_{component}_{axes.lower()}']
                delta = np.mean(np.abs((cur - prev)/prev))
                print(delta)

def fil(N, array):
    filtered = np.convolve(array, np.ones(N)/N, mode = 'valid')
    return filtered

def txt_results(df, mesh_count,fixed,number,filename=None):
    with open(filename,'a') as grfile:
        grfile.write('-'*100 + '\n')
        grfile.write(f'Grid:{number}\n')
        grfile.write(f'x = {fixed[0]}; y = {fixed[1]}\n')
        for axes in ['X','Y']:
            for component in ['XX','YY','XY']:
                grfile.write(f'{axes}:{component}\n')
                for i in range(2,mesh_count):
                    cur = df[f'mesh{i}_{component}_{axes.lower()}']
                    prev = df[f'mesh{i-1}_{component}_{axes.lower()}']
                    delta = np.mean(np.abs((cur - prev)/prev))
                    grfile.write(f'{delta}\n')

def post_processing(solver):
    os.chdir(path = f'./mesh_dir/')

    mesh_count = 7
    N = 40

    config = 'config.txt'


    sns.set_style('darkgrid')
    sns.set_palette('bright')
    colors = 'bgrcmk'
    plt.figure(figsize = (20,10))


    #For charts
    fixed_x, fixed_y = -3, -3
    x, y = np.linspace(-3,5,100), np.linspace(-4,2,100)
    x_AR, y_AR = np.linspace(-3,5,1000), np.linspace(-4,2,1000)
    #
    #
    fixed_data = pd.DataFrame()
    fixed_data['X'], fixed_data['Y'] = x, y
    AR_data = pd.DataFrame()
    AR_data['X'], AR_data['Y'] = x_AR, y_AR
    filtered_data = pd.DataFrame()
    if N % 2 == 0:
        filtered_data['X'] = x_AR[N//2-1:-N//2]
        filtered_data['Y'] = y_AR[N//2-1:-N//2]
    else:
        filtered_data['X'] = x_AR[(N-1)//2:-(N-1)//2]
        filtered_data['Y'] = y_AR[(N-1)//2:-(N-1)//2]

    for i in range(1,mesh_count):
        print(f'Mesh: {i} in process')
        path = f'./task9_{i}/'
        os.chdir(path)

        data = Data()
        data.set_params(set_default=True)
        data.make(config)

        sigma = solver.process(data)
        #solver.visual_stress(data, mises = True)

        n_elements, elements = data.elements
        n_nodes, nodes = data.nodes

        sigma_xx = [j[0][0] for j in sigma]
        sigma_yy = [j[0][1] for j in sigma]
        sigma_xy = [j[0][2] for j in sigma]
        nodes_x, nodes_y = nodes.T

        # size = (20, 10)
        #
        # plt.figure(figsize = size)
        # plt.title('Sigma X', fontsize = 15)
        # plt.tripcolor(nodes_x, nodes_y, elements, sigma_xx)
        # plt.colorbar()
        # plt.savefig('1.png')
        # plt.close()
        #
        # plt.figure(figsize = size)
        # plt.title('Sigma Y', fontsize = 15)
        # plt.tripcolor(nodes_x, nodes_y, elements, sigma_yy)
        # plt.colorbar()
        # plt.savefig('2.png')
        # plt.close()
        #
        # plt.figure(figsize = size)
        # plt.title('Sigma XY', fontsize = 15)
        # plt.tripcolor(nodes_x, nodes_y, elements, sigma_xy)
        # plt.colorbar()
        # plt.savefig('3.png')
        # plt.close()
        #
        # plt.figure(figsize = size)
        # plt.title('Sigma Mises', fontsize = 15)
        # plt.tripcolor(nodes_x, nodes_y, elements, sigma_mises)
        # plt.colorbar()
        # plt.savefig('4.png')
        # plt.close()








        #Make charts
        #fixed_x data
        fixed_data[f'mesh{i}_XX_x'] = [interpolateSigma(fixed_x, yi, sigma, 0, nodes, elements) for yi in y]
        fixed_data[f'mesh{i}_YY_x'] = [interpolateSigma(fixed_x, yi, sigma, 1, nodes, elements) for yi in y]
        fixed_data[f'mesh{i}_XY_x'] = [interpolateSigma(fixed_x, yi, sigma, 2, nodes, elements) for yi in y]

        #fixed_y data
        fixed_data[f'mesh{i}_XX_y'] = [interpolateSigma(xi, fixed_y, sigma, 0, nodes, elements) for xi in x]
        fixed_data[f'mesh{i}_YY_y'] = [interpolateSigma(xi, fixed_y, sigma, 1, nodes, elements) for xi in x]
        fixed_data[f'mesh{i}_XY_y'] = [interpolateSigma(xi, fixed_y, sigma, 2, nodes, elements) for xi in x]



        #method_AR
        #fixed_x
        res = methodAR(sigma_xx, nodes, elements)
        res_value = {i: val for i, val in enumerate(res)}
        filtered_data[f'mesh{i}_XX_x'] = fil(N,[interpolation(fixed_x,yi,res_value, nodes, elements) for yi in y_AR])

        res = methodAR(sigma_yy, nodes, elements)
        res_value = {i: val for i, val in enumerate(res)}
        filtered_data[f'mesh{i}_YY_x'] = fil(N,[interpolation(fixed_x,yi,res_value, nodes, elements) for yi in y_AR])

        res = methodAR(sigma_xy, nodes, elements)
        res_value = {i: val for i, val in enumerate(res)}
        filtered_data[f'mesh{i}_XY_x'] = fil(N,[interpolation(fixed_x,yi,res_value, nodes, elements) for yi in y_AR])

        #fixed_y
        res = methodAR(sigma_xx, nodes, elements)
        res_value = {i: val for i, val in enumerate(res)}
        filtered_data[f'mesh{i}_XX_y'] = fil(N,[interpolation(xi,fixed_y,res_value, nodes, elements) for xi in x_AR])

        res = methodAR(sigma_yy, nodes, elements)
        res_value = {i: val for i, val in enumerate(res)}
        filtered_data[f'mesh{i}_YY_y'] = fil(N,[interpolation(xi, fixed_y,res_value, nodes, elements) for xi in x_AR])

        res = methodAR(sigma_xy, nodes, elements)
        res_value = {i: val for i, val in enumerate(res)}
        filtered_data[f'mesh{i}_XY_y'] = fil(N,[interpolation(xi ,fixed_y,res_value, nodes, elements) for xi in x_AR])




        #
        # #method_AR
        # #fixed_x
        # res = methodAR(sigma_xx, nodes, elements)
        # res_value = {i: val for i, val in enumerate(res)}
        # AR_data[f'mesh{i}_XX_x'] = [interpolation(fixed_x,yi,res_value, nodes, elements) for yi in y_AR]
        #
        # res = methodAR(sigma_yy, nodes, elements)
        # res_value = {i: val for i, val in enumerate(res)}
        # AR_data[f'mesh{i}_YY_x'] = [interpolation(fixed_x,yi,res_value, nodes, elements) for yi in y_AR]
        #
        # res = methodAR(sigma_xy, nodes, elements)
        # res_value = {i: val for i, val in enumerate(res)}
        # AR_data[f'mesh{i}_XY_x'] = [interpolation(fixed_x,yi,res_value, nodes, elements) for yi in y_AR]
        #
        # #fixed_y
        # res = methodAR(sigma_xx, nodes, elements)
        # res_value = {i: val for i, val in enumerate(res)}
        # AR_data[f'mesh{i}_XX_y'] = [interpolation(xi,fixed_y,res_value, nodes, elements) for xi in x_AR]
        #
        # res = methodAR(sigma_yy, nodes, elements)
        # res_value = {i: val for i, val in enumerate(res)}
        # AR_data[f'mesh{i}_YY_y'] = [interpolation(xi, fixed_y,res_value, nodes, elements) for xi in x_AR]
        #
        # res = methodAR(sigma_xy, nodes, elements)
        # res_value = {i: val for i, val in enumerate(res)}
        # AR_data[f'mesh{i}_XY_y'] = [interpolation(xi ,fixed_y,res_value, nodes, elements) for xi in x_AR]




        # #Make filtered
        # #fixed_x data
        # filtered_data[f'mesh{i}_XX_x'] = fil(N,[interpolateSigma(fixed_x, yi, sigma, 0, nodes, elements) for yi in y])
        # filtered_data[f'mesh{i}_YY_x'] = fil(N,[interpolateSigma(fixed_x, yi, sigma, 1, nodes, elements) for yi in y])
        # filtered_data[f'mesh{i}_XY_x'] = fil(N,[interpolateSigma(fixed_x, yi, sigma, 2, nodes, elements) for yi in y])
        #
        # #fixed_y data
        # filtered_data[f'mesh{i}_XX_y'] = fil(N,[interpolateSigma(xi, fixed_y, sigma, 0, nodes, elements) for xi in x])
        # filtered_data[f'mesh{i}_YY_y'] = fil(N,[interpolateSigma(xi, fixed_y, sigma, 1, nodes, elements) for xi in x])
        # filtered_data[f'mesh{i}_XY_y'] = fil(N,[interpolateSigma(xi, fixed_y, sigma, 2, nodes, elements) for xi in x])



        os.chdir('..')

    #
    # print('Result')
    # print_results(fixed_data,mesh_count)
    # print('Filtered')
    # print_results(filtered_data,mesh_count)
    # # os.chdir('/home/ts777/fem/mesh_dir/results')
    # #
    #
    os.chdir('./results')
    #make_results(fixed_data,(fixed_x,fixed_y), mesh_count)
    #make_results(AR_data, (fixed_x,fixed_y),mesh_count ,'AR')
    make_results(filtered_data, (fixed_x,fixed_y),mesh_count ,'filteredAR')
