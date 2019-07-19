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
bat <- getNOAA.bathy(-68, -51, -51, -39, res = 1, keep = TRUE)
# dat <- as.data.frame(read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/km_test.feather'))
#head(dat)
dat <- as.data.frame(read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/km_means_full_dataset.feather'))

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
  arrange(year, month, day) %>%
  group_by(timestamp, cluster) %>%
  summarise(sum_centers = mean(sum_centers)) %>%
  group_by(timestamp) %>%
  arrange(cluster, sum_centers) %>%
  mutate(clusterS = seq_along(cluster)) %>%
  ungroup()

labels$clusterS <- labels$clusterS - 1
labels

dat <- left_join(dat, labels, by=c("timestamp"))

#dat <- as.data.frame(read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_5NN_region1_2016-03-01_2016-03-31.feather'))

dat <- select(dat, timestamp, year, month, day, hour, min, lat, lon, km_lat, km_lon, clusterS)

movdat <- dat


#movdat <- filter(movdat, NN <= NN_max)
movdat <- filter(movdat, month == "03")

#i = movdat$timestamp[100000]
i = "2016-03-15 12:00:00"
i = "2016-03-15 19:00:00"
i = "2016-03-29 00:00:00"

for (i in unique(movdat$timestamp)){
  
  #subdat <- movdat
  subdat <- filter(movdat, timestamp == i)
  subdat$clusterS <- factor(subdat$clusterS, levels = c(0, 1, 2, 3), labels = c("0", "1", "2", "3"))
  
  date_ <- paste0(subdat$year[1], "-", subdat$month[1], "-", subdat$day[1], " ", subdat$hour[1], ":", subdat$min[1], ":00")
  date_
  
  #  #C6E0FC
  
  movmap <- 
    # ggplot(bat, aes(x, y, z)) +
    autoplot(bat, geom = c("raster", "contour")) +
    geom_raster(aes(fill=z)) +
    geom_contour(aes(z = z), colour = "white", alpha = 0.05) +
    scale_fill_gradientn(values = scales::rescale(c(-6600, 0, 1, 1500)),
                         colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC", 
                                    "grey50", "grey70", "grey85")) +
    labs(x=NULL, y=NULL, color="km") +
    # geom_segment(data=subdat, aes(x=lon, xend=km_lon, y=lat, yend=km_lat, color=cluster), size=0.1) +
    geom_point(data=subdat, aes(x=lon, y=lat, color=clusterS)) +
    # geom_point(data=subdat, aes(x=km_lon, y=km_lat), color='red') +
    
    # geom_segment(data=subdat, aes(x=vessel_B_lon, xend=vessel_A_lon, y=vessel_B_lat, yend=vessel_A_lat, color=distance), size = 0.1) +
    # geom_point(data=subdat, aes(x=vessel_A_lon, y=vessel_A_lat), size = .5, color='red', alpha=0.7) +
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
    scale_color_manual(values = c("0" = "blue", "1" = "orange", "2" = "green", "3" = "black")) +
    # scale_y_continuous(expand=c(0,0)) +
    # scale_x_continuous(expand=c(0,0)) +
    NULL
  movmap
  
  print(date_)
  ggsave(filename = paste0("~/Projects/Anomalous-IUU-Events-Argentina/figures/animated_fig/hourly_figs/", date_, ".png"), width = 6, height = 4, plot = movmap)
}
