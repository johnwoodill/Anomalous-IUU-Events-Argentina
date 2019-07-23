import pandas as pd 
import numpy as np
from math import radians, cos, sin, asin, sqrt
import scipy.spatial.distance as ssd
from scipy.spatial import distance
from scipy import stats

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

def f_js(x, y):
    return distance.jensenshannon(x, y)
    #return stats.ks_2samp(x, y)

def f_ks(x, y):
    return stats.ks_2samp(x, y)

dat = pd.read_feather("~/Projects/Anomalous-IUU-Events-Argentina/data/km_means_full_dataset.feather")
dat = dat[['timestamp', 'mmsi', 'lat', 'lon', 'cluster', 'km_lon', 'km_lat']]


cent = pd.DataFrame({'cluster': [0, 1, 2, 3], 'km_lon': [-65.33, -60.58, -60.33, -58.96], 'km_lat': [-47.7, -45.9, -49.3, -41.51]})

data = dat.loc[1, :]

def get_nn(data):
    #print(dat)
    lat1 = data['lat']
    lon1 = data['lon']

    lst = []
    dst = []
    for i in range(4):
        lat2 = cent.loc[i, 'km_lat']
        lon2 = cent.loc[i, 'km_lon']
        dist = haversine(lon1, lat1, lon2, lat2)
        lst.append(dist)
    min_index = lst.index(min(lst))
    data["km_cluster"] = min_index
    data['km_dist'] = lst[min_index]
    data['km_lat'] = cent.loc[min_index, 'km_lat']
    data['km_lon'] = cent.loc[min_index, 'km_lon']
    
    return data

dat2 = dat.apply(get_nn, axis = 1)
dat2.to_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/kmedoids_full_data.feather')

dat = pd.read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/kmedoids_full_data.feather')

dat.head()

dat.loc[:, 'km_dist'] = np.log(1 + dat.loc[:, 'km_dist'])

dat.loc[:, 'month'] = pd.DatetimeIndex(dat['timestamp']).month
dat.loc[:, 'day'] = pd.DatetimeIndex(dat['timestamp']).day
dat.loc[:, 'hour'] = pd.DatetimeIndex(dat['timestamp']).hour

dat = dat.groupby(['mmsi', 'timestamp', 'month', 'day', 'hour'],
                    as_index=False)['km_dist', 'km_cluster'].mean()

dat = dat[dat.month == 3]

dat = dat.sort_values(['timestamp', 'month', 'day', 'hour'])

dat.head()

gb = dat.groupby(['month', 'day', 'hour'])
lst = [gb.get_group(x) for x in gb.groups]

rdat = pd.DataFrame()
for i in range(4):
    for j in range(len(lst) - 1):
        indat = lst[j]
        ndat = lst[j + 1]

        indat = indat[indat.km_cluster == i]
        ndat = ndat[ndat.km_cluster == i]

        p = indat['km_dist']
        q = ndat['km_dist']
        p1 = np.histogram(p, bins = 20)[0] / len(p)
        q1 = np.histogram(q, bins = 20)[0] / len(q)
        ks = f_ks(p, q)
        outdat = pd.DataFrame({'km_cluster': i, 'timestamp': indat['timestamp'].iat[0], 'jsd': [f_js(p1, q1)], 'ks_metric': ks.statistic, 'ks_p': ks.pvalue, "n_vess": len(indat['mmsi'].unique())})
        rdat = pd.concat([rdat, outdat])

rdat = rdat.reset_index()
rdat.to_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/kmedoids_jsd_results.feather')

