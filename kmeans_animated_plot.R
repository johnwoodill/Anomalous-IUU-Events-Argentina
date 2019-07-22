library(tidyverse)
library(lubridate)
library(ggmap)
library(cowplot)
library(gridExtra)
library(ggthemes)
library(feather)
library(RColorBrewer)
library(MASS)
library(plotly)
library(gg3D)
library(marmap)


#--------------------------------------------------------------
# Animated Plot

eez <- read_csv("~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_EEZ.csv")

eez <- filter(eez, lon >= -68 & lon <= -51 & lat >= -51 & lat <= -39)
eez <- filter(eez, order <= 25190)

ggplot(NULL) + geom_path(data = eez[order(eez$order), ], aes(x=lon, y=lat), linetype = "dashed")

bat <- getNOAA.bathy(-68, -51, -51, -39, res = 1, keep = TRUE)
# dat <- as.data.frame(read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/km_test.feather'))
#head(dat)
dat <- as.data.frame(read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/km_means_full_dataset.feather'))

# Subset land vessels
# !(V1 %in% c('B','N','T')
dat <- filter(dat, !(mmsi %in% c(701023000, 538002270, 701045000, 701000591, 701000578)))


dat$month <- month(dat$timestamp)
dat$day <- day(dat$timestamp)
dat$hour <- hour(dat$timestamp)
dat$year <- year(dat$timestamp)
dat$min <- minute(dat$timestamp)

dat$month <- stringr::str_pad(dat$month, 2, side = "left", pad = 0)
dat$day <- stringr::str_pad(dat$day, 2, side = "left", pad = 0)
dat$hour <- stringr::str_pad(dat$hour, 2, side = "left", pad = 0)
dat$min <- stringr::str_pad(dat$min, 2, side = "left", pad = 0)

labels <- dat %>%
  group_by(year, month, day, hour, cluster) %>%
  # summarise(sum_centers = mean(sum_centers)) %>%
  summarise(km_lat = mean(km_lat)) %>%
  group_by(year, month, day, hour) %>%
  arrange(km_lat) %>%
  mutate(clusterS = seq_along(cluster)) %>%
  select(-c(cluster)) %>% 
  arrange(year, month, day, hour) %>% 
  ungroup()

labels$clusterS <- labels$clusterS - 1
labels

dat <- left_join(dat, labels, by=c("year", "month", "day", "hour", "km_lat"))

dat$cluster <- dat$clusterS

avg_dist <- dat %>% 
  group_by(year, month, day, hour, cluster) %>% 
  summarise(km_avg_dist = round(mean(km_dist), 2))

head(avg_dist)

dat <- left_join(dat, avg_dist, by=c("year", "month", "day", "hour", "cluster"))


movdat <- dat


movdat <- filter(movdat, month == "03")

#i = movdat$timestamp[100000]
i = "2016-03-15 12:00:00"
i = "2016-03-15 19:00:00"
i = "2016-03-29 00:00:00"
i = "2016-03-02 16:00:00"


# Cluster group
cent <- data.frame(cluster = c(0, 1, 2, 3), km_lon=c(-65.33, -60.58, -60.33, -58.96), km_lat = c(-47.7, -45.9, -49.3, -41.51))


for (i in unique(movdat$timestamp)){
  
  subdat <- filter(movdat, timestamp == i)
  subdat$cluster <- factor(subdat$cluster, levels = c(0, 1, 2, 3), labels = c("0", "1", "2", "3"))
  
  subdat <- filter(subdat, lon <= -54)
  
  date_ <- paste0(subdat$year[1], "-", subdat$month[1], "-", subdat$day[1], " ", subdat$hour[1], ":", subdat$min[1], ":00")
  date_
  
  #  #C6E0FC
  
  movmap <- 
    # ggplot(bat, aes(x, y, z)) +
    autoplot(bat, geom = c("raster", "contour")) +
    geom_raster(aes(fill=z)) +
    geom_contour(aes(z = z), colour = "white", alpha = 0.05) +
    scale_fill_gradientn(values = scales::rescale(c(-6600, 1, 2, 1500)),
                         colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC", 
                                    "grey50", "grey70", "grey85")) +
    labs(x=NULL, y=NULL, color="km") +
    geom_segment(data=subdat, aes(x=lon, xend=km_lon, y=lat, yend=km_lat, color=cluster), size=0.1) +
    geom_point(data=subdat, aes(x=lon, y=lat, color=cluster)) +
    geom_point(data=subdat, aes(x=km_lon, y=km_lat), color='red') +
    geom_text(data=subdat, aes(x=km_lon, y=km_lat, label = km_avg_dist, vjust=-1), size=3.5) +
    geom_path(data = eez[order(eez$order), ], aes(x=lon, y=lat), linetype = "dashed", alpha = 0.5) +
    geom_point(data=cent, aes(km_lon, km_lat), color='green') +
    # geom_text(data=subdat, aes(x=lon, y=lat, label = mmsi, vjust=-1), size=3.5) +
    annotate("text", x=-54.5, y = -39.25, label=date_, size = 4, color='black', fontface=2) +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.direction = 'vertical',
          legend.justification = 'center',
          legend.position = "none",
          legend.margin=margin(l = 0, unit='cm'),
          legend.text = element_text(size=10),
          legend.title = element_text(size=12),
          panel.grid = element_blank()) +

    # Legend up top
    annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "black", size=1) + #Bottom
    annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Left
    annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Right
    annotate("segment", x=-Inf, xend=Inf, y=Inf, yend=Inf, color = "black", size=1) + # Top
    # scale_color_manual(values = c("0" = "blue", "1" = "orange", "2" = "green", "3" = "black")) +
    scale_color_manual(values = c("0" = "#440154FF", "1" = "#31688EFF", "2" = "#35B779FF", "3" = "#FDE725FF")) +
    # scale_y_continuous(expand=c(0,0)) +
    # scale_x_continuous(expand=c(0,0)) +
    NULL
  movmap
  
  print(date_)
  ggsave(filename = paste0("~/Projects/Anomalous-IUU-Events-Argentina/figures/animated_fig/hourly_figs/", date_, ".png"), width = 6, height = 6, plot = movmap)
}

