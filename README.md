# Anticipating and Detection Illegal Maritime Activities from Anomalous Multiscale Fleet Behavior

### James R. Watson and A. John Woodill

#### College of Earth, Ocean and Atmospheric Sciences, Oregon State University

-------------

**Repo Description**

The following repo provides reproducible results for the paper, "Anticipating and Detection Illegal Maritime Activities from Anomalous Multiscale Fleet Behavior". We utilize a K-medoids classification algorithm to disentangle a specific pre, peri, and post specific IUU event in the Puerto Madryn Argentina region. We then measure the speed of Jensen-Shannon Divergence metric to identify a rate of change of anomalous spatil behavior during the 30 day window of the IUU event. Our findings show multiscale spatial anomalies reveal a specific IUU event (cluster 2) in Puerto Madryn, Argentina. We also show the behavior of vessels is most similar during IUU events and less so outside the event window.

-------------

**Abstract**

Illegal fishing is prevalent throughout the world and heavily impacts the health of our oceans, the sustainability and profitability of fisheries, and even acts to destabilize geopolitical relations. To achieve the UN's Sustainable Development Goal of "Life Below Water", our ability to detect and predict illegal fishing must improve. Recent advances have been made through the use of vessel location data, however most analyses to date focus on anomalous spatial behaviors of vessels one at a time. To improve predictions, we use methods from complex systems and information theory to analyze the anomalous multi-scale behavior of whole fleets as they respond to nearby illegal activities. Specifically, we analyze changes in the multiscale geospatial organization of fishing fleets operating on the Patagonia Shelf, an important fishing region with chronic exposure to illegal fishing. We show that legally operating (visible) vessels respond anomalously to nearby illegal activities (which are difficult to detect). Indeed, precursor behaviors are identified, suggesting a path towards pre-empting illegal activities. This approach requires minimal data, and offers a promising step towards a global system for detecting, predicting and deterring illegal activities at sea in near real-time. Doing so will be a big step forward to achieving sustainable life under water.

**Link to paper:** [Anticipating Illegal Maritime Activities from Anomalous Multiscale Fleet Behaviors (Watson and Woodill 2019)](https://www.dropbox.com/s/gzsee168gqc81i1/Watson_Woodill_2019_AnomalousMultiscaleFleetBehaviors_CURRENT.pdf?dl=0)

-------------

**Events** 

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

<img align="center" width="500" src="https://github.com/johnwoodill/Anomalous-IUU-Events-Argentina/raw/master/figures/figure_s1.png?raw=true">

Figure S2: Dimension Reduction Results and Speed of JSD Divergence

<p align="center">

<img align="center" width="800" src="https://github.com/johnwoodill/Anomalous-IUU-Events-Argentina/raw/master/figures/figure_s2.png?raw=true">

Figure S3: Heatmap of Nearest Neighbor Distances and JSD Metric

<p align="center">

<img align="center" width="500" src="https://github.com/johnwoodill/Anomalous-IUU-Events-Argentina/raw/master/figures/figure_s3.png?raw=true">

Figure S4: Dimension Reduction Results and Speed of JSD Divergence 

<p align="center">

<img align="center" width="800" src="https://github.com/johnwoodill/Anomalous-IUU-Events-Argentina/raw/master/figures/figure_s4.png?raw=true">

Leading JS-Divergence (`t`, `t+1`)

<p align="center">

<img align="center" width="800" src="https://github.com/johnwoodill/Anomalous-IUU-Events-Argentina/raw/master/figures/figure_s5.png?raw=true">

