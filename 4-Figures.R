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

JSD <- function(p, q){
  pfun <- approxfun(density(p))
  p <- pfun(p)/sum(pfun(p))
  qfun <- approxfun(density(q))
  q <- qfun(q)/sum(qfun(q))
  m <- 0.5 * (p + q)
  JS <- 0.5 * (sum(p * log(p / m)) + sum(q * log(q / m)))
  return(JS)
}

GAPI_Key <- file("~/Projects/Anomalous-IUU-Events-Argentina/Google_api_key.txt", "r")
GAPI_Key <- readLines(GAPI_Key)
register_google(key=GAPI_Key)

#--------------------------------------------------------------------------------------------
# Figure 1: Map, NN, and distr
dat <- as.data.frame(read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_5NN_region1_2016-03-01_2016-03-31.feather'))
dat$month <- month(dat$timestamp)
dat$day <- day(dat$timestamp)
dat$hour <- hour(dat$timestamp)
dat$year <- year(dat$timestamp)
dat$min <- minute(dat$timestamp)

dat$month <- stringr::str_pad(dat$month, 2, side = "left", pad = 0)
dat$day <- stringr::str_pad(dat$day, 2, side = "left", pad = 0)
dat$hour <- stringr::str_pad(dat$hour, 2, side = "left", pad = 0)
dat$min <- stringr::str_pad(dat$min, 2, side = "left", pad = 0)

dat$ln_distance <- log(1 + dat$distance)

fig1_dat <- filter(dat, month == "03" & day == "15" & hour == "12" & NN <= 3)

# Puerto Madryn
#-42.7694° S, -65.0317° W

# Correct 4/24/2019
bat <- getNOAA.bathy(-68, -51, -51, -39, res = 1, keep = TRUE)
bat2 <- getNOAA.bathy(-77, -22, -58, -23, res = 1, keep = TRUE)

loc = c(-58, -22)
map1 <- ggmap(get_map(loc, zoom = 3, maptype='toner-background', color='bw', source='stamen')) + 
  geom_segment(x=-62, xend=-62, y=-40, yend=-45, color='red') +
  geom_segment(x=-62, xend=-54, y=-45, yend=-45, color='red') +
  geom_segment(x=-54, xend=-54, y=-45, yend=-40, color='red') +
  geom_segment(x=-62, xend=-54, y=-40, yend=-40, color='red') +
  labs(x=NULL, y=NULL) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, size=1)) +
  NULL
map1


date_ <- paste0(fig1_dat$year[1], "-", fig1_dat$month[1], "-", fig1_dat$day[1], " ", fig1_dat$hour[1], ":", fig1_dat$min[1], ":00")
date_

map2 <- 
  # ggplot(bat, aes(x, y, z)) +
  autoplot(bat, geom = c("raster", "contour")) +
  geom_raster(aes(fill=z)) +
  geom_contour(aes(z = z), colour = "white", alpha = 0.05) +
  scale_fill_gradientn(values = scales::rescale(c(-6600, 0, 1, 1500)),
                       colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC", 
                                  "grey50", "grey70", "grey85")) +
  labs(x=NULL, y=NULL, color="km") +
  geom_segment(data=fig1_dat, aes(x=vessel_A_lon, xend=vessel_B_lon, y=vessel_A_lat, yend=vessel_B_lat), size = 0.1, color='darkgrey') +
  geom_segment(data=fig1_dat, aes(x=vessel_B_lon, xend=vessel_A_lon, y=vessel_B_lat, yend=vessel_A_lat), size = 0.1, color='darkgrey') +
  geom_point(data=fig1_dat, aes(x=vessel_A_lon, y=vessel_A_lat), size = .5, color='red', alpha=0.7) +
  annotate("text", x=-51.75, y = -39.25, label="(A)", size = 4, color='black', fontface=2) +
  annotate("text", x=-65.5, y = -39.25, label=date_, size = 4, color='black', fontface=2) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.direction = 'vertical',
        legend.justification = 'center',
        legend.position = c(.95, 0.24),
        legend.margin=margin(l = 0, unit='cm'),
        legend.text = element_text(size=10),
        legend.title = element_text(size=12),
        panel.grid = element_blank()) +
  scale_color_gradientn(colours=brewer.pal(9, "OrRd"), limits=c(0, 100)) +
  
  # Legend up top
  guides(fill = FALSE,
         color = guide_colorbar(title.hjust = unit(1.1, 'cm'),
                                title.position = "top",
                                frame.colour = "black",
                                barwidth = .5,
                                barheight = 7,
                                label.position = 'left')) +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "black", size=1) + #Bottom
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Left
  annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "black", size=1) + # Right
  annotate("segment", x=-Inf, xend=Inf, y=Inf, yend=Inf, color = "black", size=1) + # Top
  # scale_y_continuous(expand=c(0,0)) +
  # scale_x_continuous(expand=c(0,0)) +
  NULL

map2


p1<-autoplot(bathy_P, geom=c("raster", "contour")) +  
  geom_point(aes(x=Long, y=Lat,color=Env),size=2, data=data_plot_PROK) + 
  geom_raster(aes(fill=z)) +
  scale_fill_gradientn(values = scales::rescale(c(-6600, 0, 1, 1500)),
                       colors = c("lightsteelblue4", "lightsteelblue2", "#C6E0FC",
                                  "grey50", "grey70", "grey85"))
  



# Figure 1b
fig1b <- ggplot(filter(fig1_dat, distance != 0), aes(log(1 + distance))) + 
  geom_histogram(aes(y=..density..), position = "dodge") +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "black") +
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "black") +
  annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "black") +
  annotate("segment", x=Inf, xend=-Inf, y=Inf, yend=Inf, color = "black") +
  theme_tufte(13) +
  annotate("text", x=6, y = 0.45, label="(B)", size = 4, color='black', fontface=2) +
  xlim(0, 6) +
  labs(y="Probability", x="Distance (log km)")
fig1b


# ggdraw() + draw_plot(map2, 0, .175, height = 1, width = 1) +
#   draw_plot(map1, .65, .368, height = .26, width = .25) +
#   draw_plot(fig1b, 0, 0, height = .385, width = 1)

# Without color guide lines
ggdraw() + draw_plot(map2, 0, .175, height = 1, width = 1) +
  draw_plot(map1, .75, .335, height = .26, width = .25) +
  draw_plot(fig1b, 0, 0, height = .385, width = 1)

ggplot2::ggsave(filename = "~/Projects/Anomalous-IUU-Events-Argentina/figures/figure1.pdf", width = 6, height = 7)
ggsave(filename = "~/Projects/Anomalous-IUU-Events-Argentina/figures/figure1.png", width = 6, height = 7)

#--------------------------------------------------------------------------------------------
# Figure 2: Distance (log transformation))

# (A)
#dat <- as.data.frame(read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_5NN_region1_2016-02-16_2016-03-16.feather'))
dat <- as.data.frame(read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_5NN_region1_2016-03-01_2016-03-31.feather'))
#dat <- as.data.frame(read_feather('~/Projects/Anomalous-IUU-Events-Argentina/data/Argentina_5NN_region1_2016-03-16_2016-04-16.feather'))

head(dat)

dat <- dat %>% 
  mutate(day = day(timestamp),
         hour = hour(timestamp),
         month = month(timestamp)) %>% 
  filter(month == 3) %>%
  filter(day >= 10 & day <= 20) %>%
  filter(distance != 0) %>% 
  group_by(timestamp, vessel_A) %>% 
  summarise(distance = mean(distance),
            ln_distance = mean(log(1 + distance)))

dat2$date <- paste0(year(dat2$timestamp), "-", month(dat2$timestamp), "-", day(dat2$timestamp), "-", hour(dat2$timestamp))

cdat <- cut(dat$ln_distance, breaks = 20)

dat$breaks <- cdat 
dat$bin <- as.numeric(dat$breaks)


char <- unique(cdat)
retdat <- data.frame()
for (j in char){
  ldat <- data.frame(a = as.numeric(strsplit(gsub("\\[|\\]|\\(|\\)", "", j), ",")[[1]][1]),
                     b = as.numeric(strsplit(gsub("\\[|\\]|\\(|\\)", "", j), ",")[[1]][2]))
  retdat <- rbind(retdat, ldat)
}

outdat <- dat %>% 
  group_by(timestamp) %>% 
  mutate(nvessels = n()) %>% 
  group_by(timestamp, bin, nvessels) %>% 
  summarise(nbin_vessels = n()) %>% 
  mutate(prob = nbin_vessels/nvessels) %>% 
  ungroup()

outdat %>% 
  group_by(timestamp) %>% 
  summarise(sum = sum(prob))

outdat$day <- day(outdat$timestamp)


outdat$timestamp
as.Date.POSIXct(outdat$timestamp, c("%Y-%m-%d %h:00:00 PST"))
outdat$timestamp <- as.POSIXct(outdat$timestamp)

sb <- c(seq(5, 50, 5))
# Get log feet to dispaly on left side
ddat <- dat %>% 
  group_by(bin) %>% 
  summarise(m_dist = mean(ln_distance)) %>% 
  filter(bin %in% sb)
ddat
ddat$x <- as.POSIXct("2016-03-09 23:00:00")

outdat <- filter(outdat, bin >= 5 & bin <= 50)
outdat$day <- day(outdat$timestamp)
a <- ifelse(outdat$day != 15, "black", "blue")



fig2a <- ggplot(outdat, aes(x=timestamp, y=factor(bin))) + 
  geom_tile(aes(fill = prob)) +
  theme_tufte(14) +
  # annotate("text", x=as.POSIXct("2016-03-20 10:00:00"), y = 45, label='(A)', size = 5, color  = "white") +
  #labs(x="Day in March", y="Binned Distance (log km)", fill="P(d)") +
  # labs(x="Day in March", y="Binned Distance", fill="P(d)") +
  geom_vline(xintercept = 14) +
  # scale_x_datetime(date_breaks = "1 day",
  #                  date_labels = "%d",
  #                  labels = c(seq(10, 21, 1)),
  #                  breaks = seq(10, 21, 1),
  #                  expand = expand_scale(mult = c(0, 0))) +
  # scale_y_discrete(breaks = c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50), 
  #                  labels = round(ddat$m_dist, 1), 
  #                  expand = c(0, 0)) +
  scale_fill_gradientn(colours=rev(brewer.pal(11, "Spectral")), na.value = 'salmon') +
  # annotate("rect", xmin = as.POSIXct("2016-03-13 01:00:00"), 
  #          xmax = as.POSIXct("2016-03-17 01:00:00"),  
  #          ymin = 0, 
  #          ymax = 47,
  #          colour="white", alpha=0.1) +
  # annotate("text", x = as.POSIXct("2016-03-15 00:00:00"), y=45, label="Event Window", color='white') +
  theme(legend.position = 'right',
        legend.margin=margin(l = 0, unit='cm'),
        panel.border = element_rect(colour = "grey", fill=NA, size=1),
        panel.grid = element_blank(),
        panel.background=element_rect(fill="#5E4FA2", colour="#5E4FA2")) +
  guides(fill = guide_colorbar(label.hjust = unit(0, 'cm'),
                               frame.colour = "black",
                               barwidth = .5,
                               barheight = 10)) +
  NULL

fig2a




# Figure 2 (B) 

dat <- as.data.frame(read_feather("~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN5_day_2016-03-01_2016-04-01.feather"))

dat$day <- seq(1, nrow(dat), 1)
pdat <- gather(dat, key = day, value = value)

pdat$day <- as.numeric(pdat$day) + 1
pdat$day2 <- seq(1, length(unique(pdat$day)))

fig2b <- ggplot(pdat, aes(x=day, y=day2)) + 
  theme_tufte(14) + 
  geom_tile(aes(fill=value)) +
  labs(y="Day in March", x="Day in March", fill="JSD \nMetric") +
  # scale_fill_gradient(low='white', high='red') +
  scale_fill_gradientn(colours=rev(brewer.pal(11, "Spectral"))) +
  scale_y_continuous(trans = "reverse", 
                     breaks = c(1, 5, 10, 15, 20, 25, 31),
                     labels = c(1, 5, 10, 15, 20, 25, 31), 
                     expand=expand_scale(mult = c(0, 0))) +
  scale_x_continuous(breaks = c(1, 5, 10, 15, 20, 25, 31), labels = c(1, 5, 10, 15, 20, 25, 31), expand=expand_scale(mult = c(0, 0))) +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "black") +
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "black") +
  annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "black") +
  annotate("segment", x=Inf, xend=-Inf, y=Inf, yend=Inf, color = "black") +
  
  # Cluster 1
  annotate("segment", x=7, xend=7, y=0.5, yend=7, color = "black") +
  annotate("segment", x=0.5, xend=7, y=7, yend=7, color = "black") +
  # 
  # # Cluster 2
  annotate("segment", x=7, xend=22, y=7, yend=7, color = "black") +
  annotate("segment", x=7, xend=7, y=7, yend=22, color = "black") +
  annotate("segment", x=22, xend=22, y=7, yend=22, color = "black") +
  annotate("segment", x=7, xend=22, y=22, yend=22, color = "black") +
  # #
  # # Cluster 3
  annotate("segment", x=22, xend=22, y=22, yend=31.5, color = "black") +
  annotate("segment", x=22, xend=31.5, y=22, yend=22, color = "black") +
  # 
  annotate("text", x=4, y = 1, label='Cluster 1', size = 3) +
  annotate("text", x=19, y = 8, label='Cluster 2', size = 3) +
  annotate("text", x=29, y = 23, label='Cluster 3', size = 3) +
  annotate("text", x=30, y = 1.5, label='(B)', size = 5, color="white") +
  guides(fill = guide_colorbar(label.hjust = unit(0, 'cm'),
                               frame.colour = "black",
                               barwidth = .5,
                               barheight = 12)) +
  coord_equal() +
  NULL
fig2b
  
ggdraw() + draw_plot(fig2a, 0, .50, height = .50, width = 1) +
  draw_plot(fig2b, 0.01, -.25, height=1, width = 1)

ggsave("~/Projects/Anomalous-IUU-Events-Argentina/figures/figure2.png", width = 5, height = 8)
ggsave("~/Projects/Anomalous-IUU-Events-Argentina/figures/figure2.pdf", width = 5, height = 8)



#--------------------------------------------------------------------------------------------
# Figure 3 - time-series of the trailing rate of change in JS divergence as an index of behavioral change

# Viridis colors
# "#440154FF" "#21908CFF" "#FDE725FF"

# rotate <- function(x) t(apply(x, 2, rev))

dat <- as.data.frame(read_feather("~/Projects/Anomalous-IUU-Events-Argentina/data/dmat_Puerto_Madryn_region1_NN5_day-hour_2016-03-01_2016-03-31.feather"))


d <- as.matrix(dat)
fit <- isoMDS(d, k=2)

# mono nMDS
#fit <- monoMDS(d, k=5, model = 'hybrid')

# Shepards plot
# d <- as.dist(d)
# shep.d <- Shepard(d, fit$points)
# plot(shep.d)


# outdat <- data.frame()
# for (i in 1:5){
#   fit <- isoMDS(d, k=i)
#   stress <- fit$stress
#   indat <- data.frame(k = i, stress = stress)
#   outdat <- rbind(outdat, indat)
# 
# }
# 
# ggplot(outdat, aes(x=k, y=stress/100)) + geom_point() + geom_line() + theme_tufte(13) + geom_hline(yintercept = 0.05, linetype = 'dashed')


stress <- fit$stress
isoMDS_dat <- data.frame(x = fit$points[, 1], y = fit$points[, 2])

# Hourly cluster
#Clusters: 

cluster1 <- c(190, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318)
cluster2 <- c(431, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 527, 534)
cluster3 <- c(651, 524, 525, 526, 528, 529, 530, 531, 532, 533, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743)

# Medoids: [190, 431, 651]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     


# Hours
isoMDS_dat$row <- seq(1, nrow(isoMDS_dat))

# Merge data for cluster factors
clustdat <- data.frame(cluster = c(rep(1, length(cluster1)),
                                   rep(2, length(cluster2)),
                                   rep(3, length(cluster3))),
                       row = c(cluster1, cluster2, cluster3))

# Merge data
isoMDS_dat <- left_join(isoMDS_dat, clustdat, by = 'row')

# Calc distance t and t+1
# 2-axis
isoMDS_dat$dist <- sqrt( (isoMDS_dat$x - lead(isoMDS_dat$x))^2 + (isoMDS_dat$y - lead(isoMDS_dat$y))^2 )

# Remove last obs
isoMDS_dat <- filter(isoMDS_dat, !is.na(cluster) | !is.na(dist))

event_day <- filter(isoMDS_dat, row >= 12*24 & row <= 17*24)
isoMDS_dat$event <- ifelse(isoMDS_dat$row >= 14*24, ifelse(isoMDS_dat$row <= 15*24, 1, 0), 0)

isoMDS_dat$speed <- isoMDS_dat$dist/1

fig3a <- ggplot(isoMDS_dat, aes(x, y, color=speed, shape=factor(cluster))) + 
  geom_point() +
  geom_point(data = filter(isoMDS_dat, event == 1), aes(x, y), color='red') +
  theme_tufte(13) +
  # xlim(-0.10, .10) +
  # ylim(-0.12, 0.12) +
  scale_color_gradientn(colours=brewer.pal(7,"YlGnBu")) +
  annotate("text", x=0.10, y = 0.10, label='(A)', size = 5, color="black") +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "grey") +
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "grey") +
  annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "grey") +
  annotate("segment", x=Inf, xend=-Inf, y=Inf, yend=Inf, color = "grey") +
  labs(x = "Dimension 1", 
       y= "Dimension 2", 
       color = "Speed", 
       shape = "Cluster") +
  theme(#aspect.ratio=1,
    legend.position = c(.5, .9),
    legend.box = "horizontal",
    legend.direction = 'horizontal',
    legend.justification = 'center',
    legend.title=element_text(size=8),
    legend.text = element_text(size=8),
    plot.title = element_text(hjust = 0.5)) +
  guides(color = guide_colorbar(order = 0, title.position = 'top', title.hjust = 0.5, barheight = .5),
         shape = guide_legend(order = 0, title.position = 'top', title.hjust = 0.5, barheight = .5)) +
  annotate("text", x = -0.085, y = -0.065, label = "k=2", size=4) +
  annotate("text", x = -0.085, y = -0.075, label = paste0("stress=", round(stress/100, 2)), size=4) +
  NULL

fig3a

  

# Color by speed (distance from day to lead(day))
fig3b <- ggplot(isoMDS_dat, aes(x=row, y=speed, color = factor(cluster))) + 
  geom_point(size=1.5, alpha = 0.7) +
  labs(x="Day in March", y="Speed of JS-Distance Divergence", color = "Cluster") +
  # annotate("text", x=24*31, y = 0.245, label='(B)', size = 5, color="black") +
  theme_tufte(13) +
  geom_vline(xintercept = 12*24, color='grey') +
  geom_vline(xintercept = 18*24, color='grey') +
  # annotate('text', x=15*24, y=0.24, label = "Event \n Window") +
  theme(legend.position = c(.10, .2),
        legend.box = "vertical",
        legend.box.background = element_rect(colour = "grey"),
        legend.direction = 'vertical',
        legend.justification = 'right',
        legend.title=element_text(size=8),
        legend.text = element_text(size=8),
        plot.title = element_text(hjust = 0.5)) +
  scale_x_continuous(breaks = c(1, 5*24, 10*24, 15*24, 20*24, 25*24, 31*24), labels = c(1, 5, 10, 15, 20, 25, 31)) +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "grey") +
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "grey") +
  annotate("segment", x=Inf, xend=Inf, y=-Inf, yend=Inf, color = "grey") +
  annotate("segment", x=Inf, xend=-Inf, y=Inf, yend=Inf, color = "grey") +
  #scale_color_viridis(discrete=TRUE) +
  scale_color_manual(breaks = c("1", "2", "3"),
                     values = c("#440154FF", "#21908CFF", "darkorange")) +
  NULL

fig3b

# Horizontal align
# ggdraw() + draw_plot(fig3a, 0, 0, height = 1, width = .5) +
#  draw_plot(fig3b, .50, 0, height= 1, width = .5)

# Verticle align
ggdraw() + draw_plot(fig3a, 0, .50, height = .5, width = 1) + 
  draw_plot(fig3b, 0, 0, height= .5, width = 1)

ggsave("~/Projects/Anomalous-IUU-Events-Argentina/figures/figure3.png", width = 8, height = 8)
ggsave("~/Projects/Anomalous-IUU-Events-Argentina/figures/figure3.pdf", width = 8, height = 8)



