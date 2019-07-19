import pandas as pd  
import numpy as np 
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import seaborn as sns
import matplotlib.pyplot as plt

#dat = pd.read_feather('data/Argentina_inter_hourly_loc_Argentina_5NN_region1_2016-02-01_2016-03-01.feather')
#dat = pd.read_feather('data/Argentina_inter_hourly_loc_Argentina_5NN_region1_2016-03-16_2016-04-16.feather')
dat = pd.read_feather('data/Argentina_inter_hourly_loc_5NN_region1_2016-03-01_2016-03-31.feather')

#dat = pd.read_feather('data/_inter_hourly_loc_Argentina_5NN_region1_2018-01-15_2018-02-15.feather')
#dat = pd.read_feather('data/_inter_hourly_loc_Argentina_5NN_region1_2018-02-05_2018-03-10.feather')

dat.dropna(inplace=True)

test = dat.groupby('mmsi', as_index=False)['lat'].agg(["max", "min"])
test.head()

#---------------------------------------
# OPTICS clustering
dat1 = dat[dat['timestamp'] == "2016-03-01 12:00:00"]

dat1 = dat[dat['timestamp'] == "2018-01-15 12:00:00"]

dat1.head()

x = dat1[['lat', 'lon']]

sns.scatterplot(x=x['lon'], y=x['lat'])

clust = OPTICS(metric="euclidean")

clust.fit(x)

# labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
#                                    core_distances=clust.core_distances_,
#                                    ordering=clust.ordering_, eps=.05)

x = dat1

# Get distances !!!
clust = OPTICS(metric="euclidean", min_cluster_size=.10)
def opt_fun(x):
    X = x[['lon', 'lat']]
    clust.fit(X)
    labels = clust.labels_[clust.ordering_]
    len_lab = len(np.unique(labels[labels != -1]))
    len_ves = len(labels[labels != -1])
    len_outlier = len(labels[labels == -1])
    m_dist = np.mean(clust.core_distances_)  # Change because has values not in cluster
    sd_dist = np.std(clust.core_distances_)  # Same

    # Build data frame with clusters and distances
    lab_dist = pd.DataFrame({'label': clust.labels_, 'dist': clust.core_distances_[clust.ordering_]})
    lab_dist = lab_dist[lab_dist['label'] != -1]
    lab_dist2 = lab_dist.groupby(['label'], as_index=False)['dist'].agg({"mean", 'std'})
    
    #len_wi_dist = np.mean(lab_dist2['mean'])
    #len_wi_sd = np.std(lab_dist2['std'])
    
    lab_dist2.loc[:, 'hour'] = x['timestamp'].iat[0]
    lab_dist2 = lab_dist2.reset_index()

    return lab_dist2

#    indat = pd.DataFrame({"hour": [x['timestamp'].iat[0]], 
#                          "nclust": len_lab, 
#                          "nves": len_ves, 
#                          "noutliers": len_outlier, 
#                          "mean_dist": m_dist, 
#                          "std_dist": sd_dist,
#                          "wi_mean_dist": len_wi_dist,
#                          "wi_std_dist": len_wi_sd})
#    return(indat)


# March 2016 Event 
dat = pd.read_feather('data/Argentina_inter_hourly_loc_5NN_region1_2016-03-01_2016-03-31.feather')
dat2 = dat.groupby('timestamp', as_index=False).apply(opt_fun)
dat2 = dat2.reset_index()
dat2.to_feather('data/cluster_temp.feather')

dat3 = dat2[(dat2['hour'] >= "2016-03-15 00:00:00") & (dat2['hour'] <= "2016-03-16 00:00:00")]
dat4 = dat2[(dat2['hour'] >= "2016-03-01 00:00:00") & (dat2['hour'] <= "2016-03-02 00:00:00")]


# -----------------------------------------------------------------
sns.set_style("white")
fig = plt.figure(figsize= [20, 12])
ax1 = fig.add_subplot(331)
ax1 = sns.scatterplot(x=range(len(dat2)), y='nclust', data=dat2, color="grey")
ax1.set(xlabel='Hours in March 2016', ylabel='Number of Clusters')
ax1.plot([14*24, 14*24], [min(dat2['nclust']), max(dat2['nclust'])], linewidth=2, color="black")
ax1.plot([15*24, 15*24], [min(dat2['nclust']), max(dat2['nclust'])], linewidth=2, color="black")

ax2 = fig.add_subplot(332)
ax2 = sns.scatterplot(x=range(len(dat2)), y='nves', data=dat2, color="grey")
ax2.set(xlabel='Hours in March 2016', ylabel='Number of Vessels')
ax2.plot([14*24, 14*24], [min(dat2['nves']), max(dat2['nves'])], linewidth=2, color="black")
ax2.plot([15*24, 15*24], [min(dat2['nves']), max(dat2['nves'])], linewidth=2, color="black")

ax3 = fig.add_subplot(333)
ax3 = sns.scatterplot(x=range(len(dat2)), y='noutliers', data=dat2, color="grey")
ax3.set(xlabel='Hours in March 2016', ylabel='Number of Outliers (no cluster)')
ax3.plot([14*24, 14*24], [min(dat2['noutliers']), max(dat2['noutliers'])], linewidth=2, color="black")
ax3.plot([15*24, 15*24], [min(dat2['noutliers']), max(dat2['noutliers'])], linewidth=2, color="black")

ax4 = fig.add_subplot(334)
ax4 = sns.scatterplot(x=range(len(dat3)), y='nclust', data=dat3, color='grey')
ax4.set(xlabel='March 15, 2016 (Day Hours)', ylabel='Number of Clusters')

ax5 = fig.add_subplot(335)
ax5 = sns.scatterplot(x=range(len(dat3)), y='nves', data=dat3, color='grey')
ax5.set(xlabel='March 15, 2016 (Day Hours)', ylabel='Number of Vessels')

ax6 = fig.add_subplot(336)
ax6 = sns.scatterplot(x=range(len(dat3)), y='noutliers', data=dat3, color='grey')
ax6.set(xlabel='March 15, 2016 (Day Hours)', ylabel='Number of Outliers')

ax7 = fig.add_subplot(337)
ax7 = sns.scatterplot(x=range(len(dat4)), y='nclust', data=dat4, color='grey')
ax7.set(xlabel='March 1, 2016 (Day Hours)', ylabel='Number of Clusters')

ax8 = fig.add_subplot(338)
ax8 = sns.scatterplot(x=range(len(dat4)), y='nves', data=dat4, color='grey')
ax8.set(xlabel='March 1, 2016 (Day Hours)', ylabel='Number of Vessels')

ax9 = fig.add_subplot(339)
ax9 = sns.scatterplot(x=range(len(dat4)), y='noutliers', data=dat4, color='grey')
ax9.set(xlabel='March 1, 2016 (Day Hours)', ylabel='Number of Outliers')

plt.show()


# Clustering data

sns.set_style("white")
fig = plt.figure(figsize= [20, 12])
ax1 = fig.add_subplot(421)
# Aggregate distance
ax1 = sns.scatterplot(x=range(len(dat2)), y='mean_dist', data=dat2, color='black')
ax1.plot([14*24, 14*24], [min(dat2['mean_dist']), max(dat2['mean_dist'])], linewidth=2, color="black")
ax1.plot([15*24, 15*24], [min(dat2['mean_dist']), max(dat2['mean_dist'])], linewidth=2, color="black")
ax1.set(xlabel='Hours in March 2016', ylabel='Aggregate Mean Distances')

ax2 = fig.add_subplot(422)
ax2 = sns.scatterplot(x=range(len(dat2)), y='std_dist', data=dat2, color='black')
ax2.plot([14*24, 14*24], [min(dat2['std_dist']), max(dat2['std_dist'])], linewidth=2, color="black")
ax2.plot([15*24, 15*24], [min(dat2['std_dist']), max(dat2['std_dist'])], linewidth=2, color="black")
ax2.set(xlabel='Hours in March 2016', ylabel='Aggregate Std. Distances')

# Within cluster distances
ax3 = fig.add_subplot(423)
ax3 = sns.scatterplot(x=range(len(dat2)), y='wi_mean_dist', data=dat2, color='black')
ax3.plot([14*24, 14*24], [min(dat2['wi_mean_dist']), max(dat2['wi_mean_dist'])], linewidth=2, color="black")
ax3.plot([15*24, 15*24], [min(dat2['wi_mean_dist']), max(dat2['wi_mean_dist'])], linewidth=2, color="black")
ax3.set(xlabel='Hours in March 2016', ylabel='Within Cluster Mean Distances')

ax4 = fig.add_subplot(424)
ax4 = sns.scatterplot(x=range(len(dat2)), y='wi_std_dist', data=dat2, color='black')
ax4.plot([14*24, 14*24], [min(dat2['wi_std_dist']), max(dat2['wi_std_dist'])], linewidth=2, color="black")
ax4.plot([15*24, 15*24], [min(dat2['wi_std_dist']), max(dat2['wi_std_dist'])], linewidth=2, color="black")
ax4.set(xlabel='Hours in March 2016', ylabel='Within Cluster Mean Distances')


ax5 = fig.add_subplot(425)
ax5 = sns.scatterplot(x=range(len(dat3)), y='mean_dist', data=dat3, color='black')
ax5.set(xlabel='Hours in March 15 2016', ylabel='Aggregate Mean Distances')


ax6 = fig.add_subplot(426)
ax6 = sns.scatterplot(x=range(len(dat3)), y='std_dist', data=dat3, color='black')
ax6.set(xlabel='Hours in March 15 2016', ylabel='Within Cluster Std. Distances')


# Within cluster distances
ax7 = fig.add_subplot(427)
ax7 = sns.scatterplot(x=range(len(dat3)), y='wi_mean_dist', data=dat3, color='black')
ax7.set(xlabel='Hours in March 15 2016', ylabel='Within Cluster Mean Distances')


ax8 = fig.add_subplot(428)
sns.scatterplot(x=range(len(dat3)), y='wi_std_dist', data=dat3, color='black')
ax8.set(xlabel='Hours in March 15 2016', ylabel='Within Cluster Std. Distances')

plt.show()


# Feb 2018 Event 
dat = pd.read_feather('data/_inter_hourly_loc_Argentina_5NN_region1_2018-01-15_2018-02-15.feather')
dat.dropna(inplace=True)
dat2 = dat.groupby('timestamp', as_index=False).apply(opt_fun)
dat3 = dat2[(dat2['hour'] >= "2018-02-02 00:00:00") & (dat2['hour'] <= "2018-02-03 00:00:00")]
dat4 = dat2[(dat2['hour'] >= "2018-01-15 00:00:00") & (dat2['hour'] <= "2018-01-16 00:00:00")]

# -----------------------------------------------------------------
sns.set_style("white")
fig = plt.figure(figsize= [20, 12])
ax1 = fig.add_subplot(331)
ax1 = sns.scatterplot(x=range(len(dat2)), y='nclust', data=dat2, color="grey")
ax1.set(xlabel='Hours in Jan-Feb 2018', ylabel='Number of Clusters')
ax1.plot([14*24, 14*24], [min(dat2['nclust']), max(dat2['nclust'])], linewidth=2, color="black")
ax1.plot([15*24, 15*24], [min(dat2['nclust']), max(dat2['nclust'])], linewidth=2, color="black")

ax2 = fig.add_subplot(332)
ax2 = sns.scatterplot(x=range(len(dat2)), y='nves', data=dat2, color="grey")
ax2.set(xlabel='Hours in Jan-Feb 2018', ylabel='Number of Vessels')
ax2.plot([14*24, 14*24], [min(dat2['nves']), max(dat2['nves'])], linewidth=2, color="black")
ax2.plot([15*24, 15*24], [min(dat2['nves']), max(dat2['nves'])], linewidth=2, color="black")

ax3 = fig.add_subplot(333)
ax3 = sns.scatterplot(x=range(len(dat2)), y='noutliers', data=dat2, color="grey")
ax3.set(xlabel='Hours in Jan-Feb 2018', ylabel='Number of Outliers (no cluster)')
ax3.plot([14*24, 14*24], [min(dat2['noutliers']), max(dat2['noutliers'])], linewidth=2, color="black")
ax3.plot([15*24, 15*24], [min(dat2['noutliers']), max(dat2['noutliers'])], linewidth=2, color="black")

ax4 = fig.add_subplot(334)
ax4 = sns.scatterplot(x=range(len(dat3)), y='nclust', data=dat3, color='grey')
ax4.set(xlabel='Feb 2, 2018 (Day Hours)', ylabel='Number of Clusters')

ax5 = fig.add_subplot(335)
ax5 = sns.scatterplot(x=range(len(dat3)), y='nves', data=dat3, color='grey')
ax5.set(xlabel='Feb 2, 2018, 2016 (Day Hours)', ylabel='Number of Vessels')

ax6 = fig.add_subplot(336)
ax6 = sns.scatterplot(x=range(len(dat3)), y='noutliers', data=dat3, color='grey')
ax6.set(xlabel='Feb 2, 2018, 2016 (Day Hours)', ylabel='Number of Outliers')

ax7 = fig.add_subplot(337)
ax7 = sns.scatterplot(x=range(len(dat4)), y='nclust', data=dat4, color='grey')
ax7.set(xlabel='Jan 15, 2016 (Day Hours)', ylabel='Number of Clusters')

ax8 = fig.add_subplot(338)
ax8 = sns.scatterplot(x=range(len(dat4)), y='nves', data=dat4, color='grey')
ax8.set(xlabel='Jan 15, 2016 (Day Hours)', ylabel='Number of Vessels')

ax9 = fig.add_subplot(339)
ax9 = sns.scatterplot(x=range(len(dat4)), y='noutliers', data=dat4, color='grey')
ax9.set(xlabel='Jan 15, 2016 (Day Hours)', ylabel='Number of Outliers')

plt.show()

# Clustering results

fig = plt.figure(figsize= [20, 12])
ax1 = fig.add_subplot(421)
# Aggregate distance
ax1 = sns.scatterplot(x=range(len(dat2)), y='mean_dist', data=dat2, color='black')
ax1.plot([18*24, 18*24], [min(dat2['mean_dist']), max(dat2['mean_dist'])], linewidth=2, color="black")
ax1.plot([19*24, 19*24], [min(dat2['mean_dist']), max(dat2['mean_dist'])], linewidth=2, color="black")
ax1.set(xlabel='Hours in Jan-Feb 2018', ylabel='Aggregate Mean Distances')

ax2 = fig.add_subplot(422)
ax2 = sns.scatterplot(x=range(len(dat2)), y='std_dist', data=dat2, color='black')
ax2.plot([18*24, 18*24], [min(dat2['std_dist']), max(dat2['std_dist'])], linewidth=2, color="black")
ax2.plot([19*24, 19*24], [min(dat2['std_dist']), max(dat2['std_dist'])], linewidth=2, color="black")
ax2.set(xlabel='Hours in Jan-Feb 2018', ylabel='Aggregate Std. Distances')

# Within cluster distances
ax3 = fig.add_subplot(423)
ax3 = sns.scatterplot(x=range(len(dat2)), y='wi_mean_dist', data=dat2, color='black')
ax3.plot([18*24, 18*24], [min(dat2['wi_mean_dist']), max(dat2['wi_mean_dist'])], linewidth=2, color="black")
ax3.plot([19*24, 19*24], [min(dat2['wi_mean_dist']), max(dat2['wi_mean_dist'])], linewidth=2, color="black")
ax3.set(xlabel='Hours in Jan-Feb 2018', ylabel='Within Cluster Mean Distances')

ax4 = fig.add_subplot(424)
ax4 = sns.scatterplot(x=range(len(dat2)), y='wi_std_dist', data=dat2, color='black')
ax4.plot([18*24, 18*24], [min(dat2['wi_std_dist']), max(dat2['wi_std_dist'])], linewidth=2, color="black")
ax4.plot([19*24, 19*24], [min(dat2['wi_std_dist']), max(dat2['wi_std_dist'])], linewidth=2, color="black")
ax4.set(xlabel='Hours in Jan-Feb 2018', ylabel='Within Cluster Mean Distances')


ax5 = fig.add_subplot(425)
ax5 = sns.scatterplot(x=range(len(dat3)), y='mean_dist', data=dat3, color='black')
ax5.set(xlabel='Hours in Feb 2, 2018', ylabel='Aggregate Mean Distances')


ax6 = fig.add_subplot(426)
ax6 = sns.scatterplot(x=range(len(dat3)), y='std_dist', data=dat3, color='black')
ax6.set(xlabel='Hours in Feb 2, 2018', ylabel='Within Cluster Std. Distances')


# Within cluster distances
ax7 = fig.add_subplot(427)
ax7 = sns.scatterplot(x=range(len(dat3)), y='wi_mean_dist', data=dat4, color='black')
ax7.set(xlabel='Hours in Feb 2, 2018', ylabel='Within Cluster Mean Distances')


ax8 = fig.add_subplot(428)
sns.scatterplot(x=range(len(dat3)), y='wi_std_dist', data=dat4, color='black')
ax8.set(xlabel='Hours in Feb 2, 2018', ylabel='Within Cluster Std. Distances')

plt.show()



# Second Feb 2018 Event 
dat = pd.read_feather('data/_inter_hourly_loc_Argentina_5NN_region1_2018-02-05_2018-03-10.feather')
dat.dropna(inplace=True)
dat2 = dat.groupby('timestamp', as_index=False).apply(opt_fun)
dat3 = dat2[(dat2['hour'] >= "2018-02-22 00:00:00") & (dat2['hour'] <= "2018-02-23 00:00:00")]
dat4 = dat2[(dat2['hour'] >= "2018-02-05 00:00:00") & (dat2['hour'] <= "2018-02-06 00:00:00")]

# -----------------------------------------------------------------
sns.set_style("white")
fig = plt.figure(figsize= [20, 12])
ax1 = fig.add_subplot(331)
ax1 = sns.scatterplot(x=range(len(dat2)), y='nclust', data=dat2, color="grey")
ax1.set(xlabel='Hours in Feb-March 2018', ylabel='Number of Clusters')
ax1.plot([17*24, 17*24], [min(dat2['nclust']), max(dat2['nclust'])], linewidth=2, color="black")
ax1.plot([18*24, 18*24], [min(dat2['nclust']), max(dat2['nclust'])], linewidth=2, color="black")

ax2 = fig.add_subplot(332)
ax2 = sns.scatterplot(x=range(len(dat2)), y='nves', data=dat2, color="grey")
ax2.set(xlabel='Hours in Feb-March 2018', ylabel='Number of Vessels')
ax2.plot([17*24, 17*24], [min(dat2['nves']), max(dat2['nves'])], linewidth=2, color="black")
ax2.plot([18*24, 18*24], [min(dat2['nves']), max(dat2['nves'])], linewidth=2, color="black")

ax3 = fig.add_subplot(333)
ax3 = sns.scatterplot(x=range(len(dat2)), y='noutliers', data=dat2, color="grey")
ax3.set(xlabel='Hours in Feb-March 2018', ylabel='Number of Outliers (no cluster)')
ax3.plot([17*24, 17*24], [min(dat2['noutliers']), max(dat2['noutliers'])], linewidth=2, color="black")
ax3.plot([18*24, 18*24], [min(dat2['noutliers']), max(dat2['noutliers'])], linewidth=2, color="black")

ax4 = fig.add_subplot(334)
ax4 = sns.scatterplot(x=range(len(dat3)), y='nclust', data=dat3, color='grey')
ax4.set(xlabel='Feb 22, 2018 (Day Hours)', ylabel='Number of Clusters')

ax5 = fig.add_subplot(335)
ax5 = sns.scatterplot(x=range(len(dat3)), y='nves', data=dat3, color='grey')
ax5.set(xlabel='Feb 22, 2018 (Day Hours)', ylabel='Number of Vessels')

ax6 = fig.add_subplot(336)
ax6 = sns.scatterplot(x=range(len(dat3)), y='noutliers', data=dat3, color='grey')
ax6.set(xlabel='Feb 22, 2018 (Day Hours)', ylabel='Number of Outliers')

ax7 = fig.add_subplot(337)
ax7 = sns.scatterplot(x=range(len(dat4)), y='nclust', data=dat4, color='grey')
ax7.set(xlabel='Feb 5, 2018 (Day Hours)', ylabel='Number of Clusters')

ax8 = fig.add_subplot(338)
ax8 = sns.scatterplot(x=range(len(dat4)), y='nves', data=dat4, color='grey')
ax8.set(xlabel='Feb 5, 2018 (Day Hours)', ylabel='Number of Vessels')

ax9 = fig.add_subplot(339)
ax9 = sns.scatterplot(x=range(len(dat4)), y='noutliers', data=dat4, color='grey')
ax9.set(xlabel='Feb 5, 2018 (Day Hours)', ylabel='Number of Outliers')

plt.show()

sns.set_style("white")
fig = plt.figure(figsize= [20, 12])
ax1 = fig.add_subplot(421)
# Aggregate distance
ax1 = sns.scatterplot(x=range(len(dat2)), y='mean_dist', data=dat2, color='black')
ax1.plot([17*24, 17*24], [min(dat2['mean_dist']), max(dat2['mean_dist'])], linewidth=2, color="black")
ax1.plot([18*24, 18*24], [min(dat2['mean_dist']), max(dat2['mean_dist'])], linewidth=2, color="black")
ax1.set(xlabel='Hours in Feb-March 2018', ylabel='Aggregate Mean Distances')

ax2 = fig.add_subplot(422)
ax2 = sns.scatterplot(x=range(len(dat2)), y='std_dist', data=dat2, color='black')
ax2.plot([17*24, 17*24], [min(dat2['std_dist']), max(dat2['std_dist'])], linewidth=2, color="black")
ax2.plot([18*24, 18*24], [min(dat2['std_dist']), max(dat2['std_dist'])], linewidth=2, color="black")
ax2.set(xlabel='Hours in Feb-March 2018', ylabel='Aggregate Std. Distances')

# Within cluster distances
ax3 = fig.add_subplot(423)
ax3 = sns.scatterplot(x=range(len(dat2)), y='wi_mean_dist', data=dat2, color='black')
ax3.plot([17*24, 17*24], [min(dat2['wi_mean_dist']), max(dat2['wi_mean_dist'])], linewidth=2, color="black")
ax3.plot([18*24, 18*24], [min(dat2['wi_mean_dist']), max(dat2['wi_mean_dist'])], linewidth=2, color="black")
ax3.set(xlabel='Hours in Feb-March 2018', ylabel='Within Cluster Mean Distances')

ax4 = fig.add_subplot(424)
ax4 = sns.scatterplot(x=range(len(dat2)), y='wi_std_dist', data=dat2, color='black')
ax4.plot([17*24, 17*24], [min(dat2['wi_std_dist']), max(dat2['wi_std_dist'])], linewidth=2, color="black")
ax4.plot([18*24, 18*24], [min(dat2['wi_std_dist']), max(dat2['wi_std_dist'])], linewidth=2, color="black")
ax4.set(xlabel='Hours in Feb-March 2018', ylabel='Within Cluster Mean Distances')


ax5 = fig.add_subplot(425)
ax5 = sns.scatterplot(x=range(len(dat3)), y='mean_dist', data=dat3, color='black')
ax5.set(xlabel='Hours in Feb 22 2018', ylabel='Aggregate Mean Distances')


ax6 = fig.add_subplot(426)
ax6 = sns.scatterplot(x=range(len(dat3)), y='std_dist', data=dat3, color='black')
ax6.set(xlabel='Hours in Feb 22 2018', ylabel='Within Cluster Std. Distances')


# Within cluster distances
ax7 = fig.add_subplot(427)
ax7 = sns.scatterplot(x=range(len(dat3)), y='wi_mean_dist', data=dat4, color='black')
ax7.set(xlabel='Hours in Feb 22 2018', ylabel='Within Cluster Mean Distances')


ax8 = fig.add_subplot(428)
sns.scatterplot(x=range(len(dat3)), y='wi_std_dist', data=dat4, color='black')
ax8.set(xlabel='Hours in Feb 22 2018', ylabel='Within Cluster Std. Distances')

plt.show()





# -------------------------------------------------------



dat2['lag_nclust'] = dat2['nclust'].shift(-1)
dat2['lag_nves'] = dat2['nves'].shift(-1)
dat2['lag_noutliers'] = dat2['noutliers'].shift(-1)

dat2['change_nclust'] = (dat2['nclust'] - dat2['lag_nclust'])/dat2['lag_nclust']
dat2['change_nves'] = (dat2['nves'] - dat2['lag_nves'])/dat2['lag_nves']
dat2['change_noutliers'] = (dat2['noutliers'] - dat2['lag_noutliers'])/dat2['lag_noutliers']


fig = plt.figure(figsize= [20, 12])
ax1 = fig.add_subplot(331)
ax1 = sns.scatterplot(x=range(len(dat2)), y='nclust', data=dat2, color="grey")
ax1.set(xlabel='Hours in March 2016', ylabel='Number of Clusters')
ax1.plot([14*24, 14*24], [min(dat2['nclust']), max(dat2['nclust'])], linewidth=2, color="black")
ax1.plot([15*24, 15*24], [min(dat2['nclust']), max(dat2['nclust'])], linewidth=2, color="black")

ax2 = fig.add_subplot(332)
ax2 = sns.scatterplot(x=range(len(dat2)), y='nves', data=dat2, color="grey")
ax2.set(xlabel='Hours in March 2016', ylabel='Number of Vessels')
ax2.plot([14*24, 14*24], [min(dat2['nves']), max(dat2['nves'])], linewidth=2, color="black")
ax2.plot([15*24, 15*24], [min(dat2['nves']), max(dat2['nves'])], linewidth=2, color="black")

ax3 = fig.add_subplot(333)
ax3 = sns.scatterplot(x=range(len(dat2)), y='noutliers', data=dat2, color="grey")
ax3.set(xlabel='Hours in March 2016', ylabel='Number of Outliers (no cluster)')
ax3.plot([14*24, 14*24], [min(dat2['noutliers']), max(dat2['noutliers'])], linewidth=2, color="black")
ax3.plot([15*24, 15*24], [min(dat2['noutliers']), max(dat2['noutliers'])], linewidth=2, color="black")

ax4 = fig.add_subplot(334)
ax4 = sns.scatterplot(x=range(len(dat3)), y='nclust', data=dat3, color='grey')
ax4.set(xlabel='March 15, 2016 (Day Hours)', ylabel='Number of Clusters')

ax5 = fig.add_subplot(335)
ax5 = sns.scatterplot(x=range(len(dat3)), y='nves', data=dat3, color='grey')
ax5.set(xlabel='March 15, 2016 (Day Hours)', ylabel='Number of Vessels')

ax6 = fig.add_subplot(336)
ax6 = sns.scatterplot(x=range(len(dat3)), y='noutliers', data=dat3, color='grey')
ax6.set(xlabel='March 15, 2016 (Day Hours)', ylabel='Number of Outliers')

ax7 = fig.add_subplot(337)
ax7 = sns.scatterplot(x=range(len(dat4)), y='nclust', data=dat4, color='grey')
ax7.set(xlabel='March 1, 2016 (Day Hours)', ylabel='Number of Clusters')

ax8 = fig.add_subplot(338)
ax8 = sns.scatterplot(x=range(len(dat4)), y='nves', data=dat4, color='grey')
ax8.set(xlabel='March 1, 2016 (Day Hours)', ylabel='Number of Vessels')

ax9 = fig.add_subplot(339)
ax9 = sns.scatterplot(x=range(len(dat4)), y='noutliers', data=dat4, color='grey')
ax9.set(xlabel='March 1, 2016 (Day Hours)', ylabel='Number of Outliers')

plt.show()


sns.scatterplot(x=range(len(dat3)), y='nves', data=dat3)

sns.scatterplot(x=range(len(dat3)), y='noutliers', data=dat3)





sns.scatterplot(x=range(len(dat2)), y='nclust', data=dat2, color="grey")
plt.plot([14*24, 14*24], [min(dat2['nclust']), max(dat2['nclust'])], linewidth=2, color="black")
plt.plot([15*24, 15*24], [min(dat2['nclust']), max(dat2['nclust'])], linewidth=2, color="black")

sns.scatterplot(x=range(len(dat2)), y='nves', data=dat2, color="grey")
plt.plot([14*24, 14*24], [min(dat2['nves']), max(dat2['nves'])], linewidth=2, color="black")
plt.plot([15*24, 15*24], [min(dat2['nves']), max(dat2['nves'])], linewidth=2, color="black")

sns.scatterplot(x=range(len(dat2)), y='noutliers', data=dat2, color="grey")
plt.plot([14*24, 14*24], [min(dat2['noutliers']), max(dat2['noutliers'])], linewidth=2, color="black")
plt.plot([15*24, 15*24], [min(dat2['noutliers']), max(dat2['noutliers'])], linewidth=2, color="black")

dat3 = dat2[(dat2['hour'] >= "2016-03-15 00:00:00") & (dat2['hour'] <= "2016-03-16 00:00:00")]

sns.scatterplot(x=range(len(dat3)), y='nclust', data=dat3)

sns.scatterplot(x=range(len(dat3)), y='nves', data=dat3)

sns.scatterplot(x=range(len(dat3)), y='noutliers', data=dat3)


