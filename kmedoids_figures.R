library(tidyverse)


dat <- read_feather("~/Projects/Anomalous-IUU-Events-Argentina/data/kmedoids_jsd_results.feather")
dat$km_cluster <- factor(dat$km_cluster, levels = c(0, 1, 2, 3), labels = c("0", "1", "2", "3"))
head(dat)

ggplot(dat, aes(x=timestamp, y=jsd, color=(km_cluster))) + 
  labs(x=NULL, y="Trailing JSD Metric") +
  theme_tufte(12) +
  geom_point() + 
  scale_color_manual(values = c("0" = "#440154FF", "1" = "#31688EFF", "2" = "#35B779FF", "3" = "#FDE725FF")) + 
  geom_vline(xintercept = as.POSIXct("2016-03-15 00:00:00")) +
  geom_vline(xintercept = as.POSIXct("2016-03-15 23:00:00")) + 
  theme(legend.position = "none",
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  facet_wrap(~km_cluster, scales = "free") 

ggplot(dat, aes(x=timestamp, y=n_vess, color=(km_cluster))) + 
  labs(x=NULL, y="# Vessels") +
  theme_tufte(12) +
  geom_line() + 
  scale_color_manual(values = c("0" = "#440154FF", "1" = "#31688EFF", "2" = "#35B779FF", "3" = "#FDE725FF")) + 
  geom_vline(xintercept = as.POSIXct("2016-03-15 00:00:00")) +
  geom_vline(xintercept = as.POSIXct("2016-03-15 23:00:00")) + 
  theme(legend.position = "none",
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  facet_wrap(~km_cluster, scales = "free") 

ggplot(dat, aes(x=timestamp, y=ks_metric, color=(km_cluster))) + 
  labs(x=NULL, y="Trailing KS Statistic") +
  theme_tufte(12) +
  geom_line() + 
  scale_color_manual(values = c("0" = "#440154FF", "1" = "#31688EFF", "2" = "#35B779FF", "3" = "#FDE725FF")) + 
  geom_vline(xintercept = as.POSIXct("2016-03-15 00:00:00")) +
  geom_vline(xintercept = as.POSIXct("2016-03-15 23:00:00")) + 
  theme(legend.position = "none",
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  facet_wrap(~km_cluster, scales = "free") 

ggplot(dat, aes(x=timestamp, y=ks_p, color=(km_cluster))) + 
  labs(x=NULL, y="KS p-value") +
  theme_tufte(12) +
  geom_line() + 
  scale_color_manual(values = c("0" = "#440154FF", "1" = "#31688EFF", "2" = "#35B779FF", "3" = "#FDE725FF")) + 
  geom_vline(xintercept = as.POSIXct("2016-03-15 00:00:00")) +
  geom_vline(xintercept = as.POSIXct("2016-03-15 23:00:00")) + 
  theme(legend.position = "none",
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  facet_wrap(~km_cluster, scales = "free") 
