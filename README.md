# Anticipating and Detection Illegal Maritime Activities from Anomalous Multiscale Fleet Behavior

### James R. Watson and A. John Woodill

#### College of Earth, Ocean and Atmospheric Sciences, Oregon State University

-------------

The following repo provides reproducible results for the paper, "Anticipating and Detection Illegal Maritime Activities from Anomalous Multiscale Fleet Behavior". We utilize a novel  K-medoids classification algorithm to disentangle a specific pre, peri, and post specific IUU event in the Puerto Madryn Argentina region. Our main findings show multiscale spatial anomalies reveal a specific IUU event (cluster 2) in Puerto Madryn, Argentina. We also show the behavior of vessels is most similar during IUU events and less so outside the event window.

**Events:** 

March 15, 2016

https://www.cnn.com/2016/03/15/americas/argentina-chinese-fishing-vessel/index.html


Feb 2, 2018

http://www.laht.com/article.asp?CategoryId=14093&ArticleId=2450374


Feb 21, 2018

https://www.reuters.com/article/us-argentina-china-fishing/argentina-calls-for-capture-of-five-chinese-fishing-boats-idUSKCN1GK35T


--------------

**Folders**

* `data/` - data available for release

* `figures/` - main figures

* `spatialIUU/` - custom classes and methods. See [spatialIUU repo](https://github.com/johnwoodill/spatialIUU) for additional information and build tests.


--------------

**Files**

* `1-Data-step.py` - process pre-processed Global Fishing Watch (GFW) AIS data (not available).

* `2-Cluster-Analysis.py` - K-medoids clustering analysis for all events.

* `4-Figures.R` - main figures from the analysis.

* `5-Supp_figures.R` - supplement figures for additional events.


-------------



### **Figures**

Figure 1: Map of Patagonia Shelf and Distribution of Nearest Neighbor Distances
<p align="center">

<img align="center" width="500" src="https://github.com/johnwoodill/Anomalous-IUU-Events-Argentina/raw/master/figures/figure1.png?raw=true">

Figure 2: Heatmap of Nearest Neighbor Distances and JSD Metric

<p align="center">

<img align="center" width="500" src="https://github.com/johnwoodill/Anomalous-IUU-Events-Argentina/raw/master/figures/figure2.png?raw=true">


Figure 3: Dimension Reduction Results and Speed of JSD Divergence

<p align="center">

<img align="center" width="800" src="https://github.com/johnwoodill/Anomalous-IUU-Events-Argentina/raw/master/figures/figure3.png?raw=true">

-------------

### *Supporting Figures*

Figure S1: Heatmap of Nearest Neighbor Distances and JSD Metric

<p align="center">

<img align="center" width="800" src="https://github.com/johnwoodill/Anomalous-IUU-Events-Argentina/raw/master/figures/figure_s1.png?raw=true">

Figure S2: Dimension Reduction Results and Speed of JSD Divergence

<p align="center">

<img align="center" width="800" src="https://github.com/johnwoodill/Anomalous-IUU-Events-Argentina/raw/master/figures/figure_s2.png?raw=true">

Figure S3: Heatmap of Nearest Neighbor Distances and JSD Metric

<p align="center">

<img align="center" width="800" src="https://github.com/johnwoodill/Anomalous-IUU-Events-Argentina/raw/master/figures/figure_s3.png?raw=true">

Figure S4: Dimension Reduction Results and Speed of JSD Divergence 

<p align="center">

<img align="center" width="800" src="https://github.com/johnwoodill/Anomalous-IUU-Events-Argentina/raw/master/figures/figure_s4.png?raw=true">

Leading JS-Divergence (`t`, `t+1`)

<p align="center">

<img align="center" width="500" src="https://github.com/johnwoodill/Puerto_Madryn_IUU_Fleet_Behavior/raw/master/figures/supporting/leading_JS.png?raw=true">

