import pandas as pd  
import numpy as np 
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import seaborn as sns
import matplotlib.pyplot as plt
dat = pd.read_feather('data/_inter_hourly_loc_Argentina_5NN_region1_2016-02-01_2016-03-01.feather')
dat = pd.read_feather('data/_inter_hourly_loc_Argentina_5NN_region1_2016-03-16_2016-04-16.feather')
dat = pd.read_feather('data/_inter_hourly_loc_Argentina_5NN_region1_2016-03-01_2016-03-31.feather')


#---------------------------------------
# OPTICS clustering
dat1 = dat[dat['timestamp'] == "2016-03-01 12:00:00"]

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
clust = OPTICS(metric="euclidean")
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
    
    len_wi_dist = np.mean(lab_dist2['mean'])
    len_wi_sd = np.std(lab_dist2['std'])
    
    indat = pd.DataFrame({"hour": [x['timestamp'].iat[0]], 
                          "nclust": len_lab, 
                          "nves": len_ves, 
                          "noutliers": len_outlier, 
                          "mean_dist": m_dist, 
                          "std_dist": sd_dist,
                          "wi_mean_dist": len_wi_dist,
                          "wi_std_dist": len_wi_sd})
    return(indat)


dat2 = dat.groupby('timestamp', as_index=False).apply(opt_fun)

dat3 = dat2[(dat2['hour'] >= "2016-03-15 00:00:00") & (dat2['hour'] <= "2016-03-16 00:00:00")]
dat4 = dat2[(dat2['hour'] >= "2016-03-01 00:00:00") & (dat2['hour'] <= "2016-03-02 00:00:00")]

# Aggregate distance
sns.scatterplot(x=range(len(dat3)), y='mean_dist', data=dat3, color='black')
plt.plot([14*24, 14*24], [min(dat3['mean_dist']), max(dat3['mean_dist'])], linewidth=2, color="black")
plt.plot([15*24, 15*24], [min(dat3['mean_dist']), max(dat3['mean_dist'])], linewidth=2, color="black")
plt.xlabel("Hours in March 2016")
plt.ylabel("Aggregate Mean Distances")

sns.scatterplot(x=range(len(dat2)), y='std_dist', data=dat2, color='black')
plt.plot([14*24, 14*24], [min(dat2['std_dist']), max(dat2['std_dist'])], linewidth=2, color="black")
plt.plot([15*24, 15*24], [min(dat2['std_dist']), max(dat2['std_dist'])], linewidth=2, color="black")
plt.xlabel("Hours in March 2016")
plt.ylabel("Aggregate Mean Distances")

# Within cluster distances
sns.scatterplot(x=range(len(dat2)), y='wi_mean_dist', data=dat2, color='black')
plt.plot([14*24, 14*24], [min(dat2['wi_mean_dist']), max(dat2['wi_mean_dist'])], linewidth=2, color="black")
plt.plot([15*24, 15*24], [min(dat2['wi_mean_dist']), max(dat2['wi_mean_dist'])], linewidth=2, color="black")
plt.xlabel("Hours in March 2016")
plt.ylabel("Within Cluster Mean Distances")

sns.scatterplot(x=range(len(dat2)), y='wi_std_dist', data=dat2, color='black')
plt.plot([14*24, 14*24], [min(dat2['wi_std_dist']), max(dat2['wi_std_dist'])], linewidth=2, color="black")
plt.plot([15*24, 15*24], [min(dat2['wi_std_dist']), max(dat2['wi_std_dist'])], linewidth=2, color="black")
plt.xlabel("Hours in March 2016")
plt.ylabel("Within Cluster Mean Distances")




dat2['lag_nclust'] = dat2['nclust'].shift(-1)
dat2['lag_nves'] = dat2['nves'].shift(-1)
dat2['lag_noutliers'] = dat2['noutliers'].shift(-1)

dat2['change_nclust'] = (dat2['nclust'] - dat2['lag_nclust'])/dat2['lag_nclust']
dat2['change_nves'] = (dat2['nves'] - dat2['lag_nves'])/dat2['lag_nves']
dat2['change_noutliers'] = (dat2['noutliers'] - dat2['lag_noutliers'])/dat2['lag_noutliers']


dat3 = dat2[(dat2['hour'] >= "2016-03-15 00:00:00") & (dat2['hour'] <= "2016-03-16 00:00:00")]
dat4 = dat2[(dat2['hour'] >= "2016-03-01 00:00:00") & (dat2['hour'] <= "2016-03-02 00:00:00")]

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


#-------------------------------------------------
# DBSCAN



print(dat2)




print(mdat)

for i in dat.timestamp.unique:
    print(i)
    


np.unique(clust.labels_)
len(np.unique(clust.labels_))

dat1['cluster'] = clust.labels_

pdat = dat1[dat1.cluster != -1]
sns.scatterplot(x='lon', y='lat', hue='cluster', data = pdat, s=10)







np.unique(labels_050)

np.unique(labels_200)

len(labels_050)
len(dat1)

space = np.arange(len(x))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Reachability plot
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

# OPTICS
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = X[clust.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
ax2.set_title('Automatic Clustering\nOPTICS')

# DBSCAN at 0.5
colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
for klass, color in zip(range(0, 6), colors):
    Xk = X[labels_050 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

# DBSCAN at 2.
colors = ['g.', 'm.', 'y.', 'c.']
for klass, color in zip(range(0, 4), colors):
    Xk = X[labels_200 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

plt.tight_layout()
plt.show()






    savedat = outdat.reset_index(drop=True)
    savedat.to_feather(f"{PROC_DATA_LOC}_inter_hourly_loc_{REGION}_5NN_region{region}_{beg_date}_{end_date}.feather")
