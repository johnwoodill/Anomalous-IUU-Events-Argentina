import pandas as pd
from scipy import stats
import numpy as np
from dit.divergences import jensen_shannon_divergence
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns
import scipy.spatial.distance as ssd
import random
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer, cluster_visualizer_multidim
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import read_sample
from pyclustering.utils import timedcall
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.spatial.distance as ssd

import spatialIUU.distanceMatrix as dm

def k_medoids(distMatrix, interval, init_medoids):
    distMatrix = np.array(distMatrix)

    # K-Medoids Clustering
    #initial_medoids = [30, 90, 140]

    initial_medoids = init_medoids
    # create K-Medoids algorithm for processing distance matrix instead of points
    kmedoids_instance = kmedoids(
        distMatrix, initial_medoids, data_type='distance_matrix', ccore=True)

    # run cluster analysis and obtain results
    kmedoids_instance.process()

    clusters = kmedoids_instance.get_clusters()
    medoids = kmedoids_instance.get_medoids()
    print(f"Clusters: {clusters}   Medoids: {medoids}")

    final_list = []
    for i, l in enumerate(clusters):
        for num in l:
            final_list.append({'value': num, 'group': i})

    df = pd.DataFrame(final_list)
    return(df)


print("[1/4] Calculating distance matrices March 1-31 2016")
# ------------------------------------------------------
# Puerto Madryn March 1-31 2016 Region 1
## NN = 1

# Import data
dat = pd.read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_5NN_region1_2016-03-01_2016-03-31.feather')

# Day
# NN = 1
distMatrix, distArray = dm.d_matrix(dat, interval='day', NN=1)
pdat1 = k_medoids(distMatrix, interval='day', init_medoids=[2, 5, 8])

# Convert matrix to data.frame
distMatrix = pd.DataFrame(distMatrix)
distMatrix.columns = distMatrix.columns.astype(str)

# Save file
distMatrix.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN1_day_2016-03-01_2016-04-01.feather')


# NN = 1
distMatrix, distArray = dm.d_matrix(dat, interval='day', NN=5)
pdat1 = k_medoids(distMatrix, interval='day', init_medoids=[2, 5, 8])

# Convert matrix to data.frame
distMatrix = pd.DataFrame(distMatrix)
distMatrix.columns = distMatrix.columns.astype(str)

# Save file
distMatrix.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN5_day_2016-03-01_2016-04-01.feather')


# Day by hour
# NN = 1
distMatrix_dh1, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=1)
#pdat2 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[144, 360, 480])

distMatrix_dh1 = pd.DataFrame(distMatrix_dh1)
distMatrix_dh1.columns = distMatrix_dh1.columns.astype(str)

# Save distance matrix
distMatrix_dh1.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN1_day-hour_2016-03-01_2016-03-31.feather')

np.save('~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN1_day-hour_2016-03-01_2016-03-31.npy')

distMatrix_dh2, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=2)
#pdat2 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[144, 360, 480])

distMatrix_dh2 = pd.DataFrame(distMatrix_dh2)
distMatrix_dh2.columns = distMatrix_dh2.columns.astype(str)

# Save distance matrix
distMatrix_dh2.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN2_day-hour_2016-03-01_2016-03-31.feather')


distMatrix_dh3, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=3)
#pdat2 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[144, 360, 480])

distMatrix_dh3 = pd.DataFrame(distMatrix_dh3)
distMatrix_dh3.columns = distMatrix_dh3.columns.astype(str)

# Save distance matrix
distMatrix_dh3.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN3_day-hour_2016-03-01_2016-03-31.feather')


distMatrix_dh4, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=4)
#pdat2 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[144, 360, 480])

distMatrix_dh4 = pd.DataFrame(distMatrix_dh4)
distMatrix_dh4.columns = distMatrix_dh4.columns.astype(str)

# Save distance matrix
distMatrix_dh4.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN4_day-hour_2016-03-01_2016-03-31.feather')


distMatrix_dh5, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=5)
#pdat2 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[144, 360, 480])

distMatrix_dh5 = pd.DataFrame(distMatrix_dh5)
distMatrix_dh5.columns = distMatrix_dh5.columns.astype(str)

# Save distance matrix
distMatrix_dh5.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN5_day-hour_2016-03-01_2016-03-31.feather')


print("[2/4] Calculating distance matrices Jan 15 - Feb 15 2018")
# ------------------------------------------------------
# Puerto Madryn January 15-February 15 2018 Region 1
# Import data
dat = pd.read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_5NN_region1_2018-01-15_2018-02-15.feather')

# Day
# NN = 1
distMatrix, distArray = dm.d_matrix(dat, interval='day', NN=1)
#pdat3 = k_medoids(distMatrix, interval='day', init_medoids=[7, 18, 25])

distMatrix = pd.DataFrame(distMatrix)
distMatrix.columns = distMatrix.columns.astype(str)

distMatrix.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN1_day_2018-01-15_2018-02-15.feather')


distMatrix, distArray = dm.d_matrix(dat, interval='day', NN=5)
#pdat3 = k_medoids(distMatrix, interval='day', init_medoids=[7, 18, 25])

distMatrix = pd.DataFrame(distMatrix)
distMatrix.columns = distMatrix.columns.astype(str)

distMatrix.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN5_day_2018-01-15_2018-02-15.feather')




# Day by hour
# NN = 1
distMatrix_dh6, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=1)
#pdat4 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[168, 432, 600])

# Convert matrix to data.frame
distMatrix_dh6 = pd.DataFrame(distMatrix_dh6)
distMatrix_dh6.columns = distMatrix_dh6.columns.astype(str)

# Save distance matrix
distMatrix_dh6.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN1_day-hour_2018-01-15_2018-02-15.feather')

distMatrix_dh7, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=2)
#pdat4 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[168, 432, 600])

# Convert matrix to data.frame
distMatrix_dh7 = pd.DataFrame(distMatrix_dh7)
distMatrix_dh7.columns = distMatrix_dh7.columns.astype(str)

# Save distance matrix
distMatrix_dh7.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN2_day-hour_2018-01-15_2018-02-15.feather')


distMatrix_dh8, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=3)
#pdat4 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[168, 432, 600])

# Convert matrix to data.frame
distMatrix_dh8 = pd.DataFrame(distMatrix_dh8)
distMatrix_dh8.columns = distMatrix_dh8.columns.astype(str)

# Save distance matrix
distMatrix_dh8.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN3_day-hour_2018-01-15_2018-02-15.feather')


distMatrix_dh9, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=4)
#pdat4 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[168, 432, 600])

# Convert matrix to data.frame
distMatrix_dh9 = pd.DataFrame(distMatrix_dh9)
distMatrix_dh9.columns = distMatrix_dh9.columns.astype(str)

# Save distance matrix
distMatrix_dh9.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN4_day-hour_2018-01-15_2018-02-15.feather')


distMatrix_dh10, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=5)
#pdat4 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[168, 432, 600])

# Convert matrix to data.frame
distMatrix_dh10 = pd.DataFrame(distMatrix_dh10)
distMatrix_dh10.columns = distMatrix_dh10.columns.astype(str)

# Save distance matrix
distMatrix_dh10.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN5_day-hour_2018-01-15_2018-02-15.feather')


print("[3/4] Calculating distance matrices Feb 5 - March 10 2018")
# ------------------------------------------------------
# Puerto Madryn February 05 to March 10 2018 Region 1
# Import data
dat = pd.read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_5NN_region1_2018-02-05_2018-03-10.feather')

# Day
distMatrix, distArray = dm.d_matrix(dat, interval='day', NN=1)
#pdat5 = k_medoids(distMatrix, interval='day', init_medoids=[7, 14, 21])

distMatrix = pd.DataFrame(distMatrix)
distMatrix.columns = distMatrix.columns.astype(str)


distMatrix.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN1_day_2018-02-05_2018-03-10.feather')


# Day
distMatrix, distArray = dm.d_matrix(dat, interval='day', NN=5)
#pdat5 = k_medoids(distMatrix, interval='day', init_medoids=[7, 14, 21])

distMatrix = pd.DataFrame(distMatrix)
distMatrix.columns = distMatrix.columns.astype(str)


distMatrix.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN5_day_2018-02-05_2018-03-10.feather')


# Day by hour
# NN = 1
distMatrix_dh11, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=1)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh11 = pd.DataFrame(distMatrix_dh11)
distMatrix_dh11.columns = distMatrix_dh11.columns.astype(str)

# Save distance matrix
distMatrix_dh11.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN1_day-hour_2018-02-05_2018-03-10.feather')    


distMatrix_dh12, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=2)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh12 = pd.DataFrame(distMatrix_dh12)
distMatrix_dh12.columns = distMatrix_dh12.columns.astype(str)

# Save distance matrix
distMatrix_dh12.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN2_day-hour_2018-02-05_2018-03-10.feather')    


distMatrix_dh13, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=3)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh13 = pd.DataFrame(distMatrix_dh13)
distMatrix_dh13.columns = distMatrix_dh13.columns.astype(str)

# Save distance matrix
distMatrix_dh13.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN3_day-hour_2018-02-05_2018-03-10.feather')    


distMatrix_dh14, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=4)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh14 = pd.DataFrame(distMatrix_dh14)
distMatrix_dh14.columns = distMatrix_dh14.columns.astype(str)

# Save distance matrix
distMatrix_dh14.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN4_day-hour_2018-02-05_2018-03-10.feather')    


distMatrix_dh15, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=5)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh15 = pd.DataFrame(distMatrix_dh15)
distMatrix_dh15.columns = distMatrix_dh15.columns.astype(str)

# Save distance matrix
distMatrix_dh15.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN5_day-hour_2018-02-05_2018-03-10.feather')    


# Two weeks prior, one week after
print("[4/4] Calculating distance matrices March 1 - 14")
# ------------------------------------------------------
# Puerto Madryn March 1-31 2016 Region 1
## NN = 1

# Import data
dat = pd.read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_5NN_region1_2016-03-01_2016-03-31.feather')

# Subset March 1 - 14
dat1 = dat[(dat.timestamp >= f" 03-01-2016 00:00:00") & (dat.timestamp <= f"03-14-2016 23:59:00")]

distMatrix_dh16, distArray_dh = dm.d_matrix(dat1, interval='dayhour', NN=1)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh16 = pd.DataFrame(distMatrix_dh16)
distMatrix_dh16.columns = distMatrix_dh16.columns.astype(str)

# Save distance matrix
distMatrix_dh16.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN1_day-hour_2016-03-01_2016-03-14.feather')    


distMatrix_dh17, distArray_dh = dm.d_matrix(dat1, interval='dayhour', NN=5)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh17 = pd.DataFrame(distMatrix_dh17)
distMatrix_dh17.columns = distMatrix_dh17.columns.astype(str)

# Save distance matrix
distMatrix_dh17.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN5_day-hour_2016-03-01_2016-03-14.feather')    


# Week after
dat2 = dat[(dat.timestamp >= f" 03-22-2016 00:00:00") & (dat.timestamp <= f"03-31-2016 23:59:00")]

distMatrix_dh18, distArray_dh = dm.d_matrix(dat2, interval='dayhour', NN=1)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh18 = pd.DataFrame(distMatrix_dh18)
distMatrix_dh18.columns = distMatrix_dh18.columns.astype(str)

# Save distance matrix
distMatrix_dh18.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN1_day-hour_2016-03-22_2016-03-31.feather')    


distMatrix_dh19, distArray_dh = dm.d_matrix(dat2, interval='dayhour', NN=5)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh19 = pd.DataFrame(distMatrix_dh19)
distMatrix_dh19.columns = distMatrix_dh19.columns.astype(str)

# Save distance matrix
distMatrix_dh19.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN5_day-hour_2016-03-22_2016-03-31.feather')    

# April 2016 Non-event
print("[5/5] Calculating distance matrices April 1 - 30 2016")
# ------------------------------------------------------

# Import data
dat = pd.read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_5NN_region1_2016-04-01_2016-04-30.feather')


distMatrix_dh20, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=1)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh20 = pd.DataFrame(distMatrix_dh20)
distMatrix_dh20.columns = distMatrix_dh20.columns.astype(str)

# Save distance matrix
distMatrix_dh20.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN1_day-hour_2016-04-01_2016-04-30.feather')    


distMatrix_dh21, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=5)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh21 = pd.DataFrame(distMatrix_dh21)
distMatrix_dh21.columns = distMatrix_dh21.columns.astype(str)

# Save distance matrix
distMatrix_dh21.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN5_day-hour_2016-04-01_2016-04-30.feather')    



# April 2016 Non-event
print("[5/5] Calculating distance matrices April 1 - 30 2016")
# ------------------------------------------------------

# Import data
dat = pd.read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_5NN_region1_2016-04-15_2016-04-30.feather')


distMatrix_dh20, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=1)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh20 = pd.DataFrame(distMatrix_dh20)
distMatrix_dh20.columns = distMatrix_dh20.columns.astype(str)

# Save distance matrix
distMatrix_dh20.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN1_day-hour_2016-04-01_2016-04-30.feather')    


distMatrix_dh21, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=5)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh21 = pd.DataFrame(distMatrix_dh21)
distMatrix_dh21.columns = distMatrix_dh21.columns.astype(str)

# Save distance matrix
distMatrix_dh21.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN5_day-hour_2016-04-01_2016-04-30.feather')    


# April 2016 Non-event
print("[6/6] Calculating distance matrices April 15 - May 15 2016")
# ------------------------------------------------------

# Import data
dat = pd.read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_5NN_region1_2016-04-15_2016-05-15.feather')


distMatrix_dh22, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=1)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh22 = pd.DataFrame(distMatrix_dh22)
distMatrix_dh22.columns = distMatrix_dh22.columns.astype(str)

# Save distance matrix
distMatrix_dh22.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN1_day-hour_2016-04-15_2016-05-15.feather')    


distMatrix_dh23, distArray_dh = dm.d_matrix(dat, interval='dayhour', NN=5)
#pdat6 = k_medoids(distMatrix_dh, interval='dayhour',
#                  init_medoids=[156, 324, 492])

# Convert matrix to data.frame
distMatrix_dh23 = pd.DataFrame(distMatrix_dh23)
distMatrix_dh23.columns = distMatrix_dh23.columns.astype(str)

# Save distance matrix
distMatrix_dh23.to_feather(
    '~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN5_day-hour_2016-04-15_2016-05-15.feather')    
