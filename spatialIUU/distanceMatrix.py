import pandas as pd
import numpy as np
import scipy.spatial.distance as ssd
from scipy.spatial import distance

def f_js(x, y):
    return distance.jensenshannon(x, y)


def d_matrix(dat, interval, NN=0):
    dat = dat[dat['NN'] != 0]
    dat = dat[dat['NN'] <= NN]
    dat.loc[:, 'distance'] = np.log(1 + dat.loc[:, 'distance'])

    dat.loc[:, 'month'] = pd.DatetimeIndex(dat['timestamp']).month
    dat.loc[:, 'day'] = pd.DatetimeIndex(dat['timestamp']).day
    dat.loc[:, 'hour'] = pd.DatetimeIndex(dat['timestamp']).hour

    if interval == 'day':
        dat = dat.groupby(['vessel_A', 'month', 'day'], as_index=False)[
            'distance'].mean()
        x = []

        gb = dat.groupby(['month', 'day'])['distance']
        lst = [gb.get_group(x) for x in gb.groups]
        x = []
        for i in range(len(lst)):
            for j in range(len(lst)):
                x += [(i, j, f_js(lst[i], lst[j]))]

        distMatrix = pd.DataFrame(x).pivot(index=0, columns=1, values=2)
        distMatrix = np.matrix(distMatrix)
        distArray = ssd.squareform(distMatrix)

    if interval == 'dayhour':
        dat = dat.groupby(['vessel_A', 'month', 'day', 'hour'],
                          as_index=False)['distance'].mean()
        x = []

        gb = dat.groupby(['month', 'day', 'hour'])['distance']
        lst = [gb.get_group(x) for x in gb.groups]
        x = []
        for i in range(len(lst)):
            for j in range(len(lst)):
                x += [(i, j, f_js(lst[i], lst[j]))]

        distMatrix = pd.DataFrame(x).pivot(index=0, columns=1, values=2)
        distMatrix = np.matrix(distMatrix)
        distArray = ssd.squareform(distMatrix)

    return (distMatrix, distArray)

