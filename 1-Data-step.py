import spatialIUU.processGFW as siuu
import os

# Set global constants
global GFW_DIR, GFW_OUT_DIR_CSV, GFW_OUT_DIR_FEATHER, PROC_DATA_LOC, MAX_SPEED, REGION, lon1, lon2, lat1, lat2

siuu.GFW_DIR = '/data2/GFW_point/'
siuu.GFW_OUT_DIR_CSV = '/home/server/pi/homes/woodilla/Data/GFW_point/Patagonia_Shelf/csv/'
siuu.GFW_OUT_DIR_FEATHER = '/home/server/pi/homes/woodilla/Data/GFW_point/Patagonia_Shelf/feather/'
siuu.PROC_DATA_LOC = '/home/server/pi/homes/woodilla/Projects/Anomalous-IUU-Events-Argentina/data/'
siuu.REGION = 'Argentina'
siuu.MAX_SPEED = 32

# Check if dir exists and create
os.makedirs(siuu.PROC_DATA_LOC, exist_ok=True) 

siuu.region = 1
siuu.lon1 = -68
siuu.lon2 = -54
siuu.lat1 = -51
siuu.lat2 = -39


# First event: 
# https://www.cnn.com/2016/03/15/americas/argentina-chinese-fishing-vessel/index.html
siuu.compileData('2016-03-01', '2016-03-31', 1, parallel=True, ncores=20)


# Second event: Feb 2, 2018
# http://www.laht.com/article.asp?CategoryId=14093&ArticleId=2450374
#siuu.compileData('2018-01-15', '2018-02-15', 1, parallel=True, ncores=20)


# Third event: Feb 21, 2018
# https://www.reuters.com/article/us-argentina-china-fishing/argentina-calls-for-capture-of-five-chinese-fishing-boats-idUSKCN1GK35T
#siuu.compileData('2018-02-05', '2018-03-10', 1, parallel=True, ncores=20)

# Robust check for month preceeding event
#siuu.compileData('2016-04-01', '2016-04-30', 1, parallel=True, ncores=20)

# Robust check for month preceeding event
#siuu.compileData('2016-04-15', '2016-05-15', 1, parallel=True, ncores=20)

# Predict event is happening
#siuu.compileData('2016-02-16', '2016-03-16', 1, parallel=True, ncores=20)

# Second event: Feb 2, 2018 preceeding
#siuu.compileData('2018-01-02', '2018-02-02', 1, parallel=True, ncores=20)

# Month after IUU
#siuu.compileData('2016-03-16', '2016-04-16', 1, parallel=True, ncores=20)

# Month before IUU
#siuu.compileData('2016-02-01', '2016-03-01', 1, parallel=True, ncores=30)


# Third event: Feb 21, 2018
# https://www.reuters.com/article/us-argentina-china-fishing/argentina-calls-for-capture-of-five-chinese-fishing-boats-idUSKCN1GK35T
#siuu.compileData('2018-01-21', '2018-02-21', 1, parallel=True, ncores=20)

# Month after IUU
#siuu.compileData('2016-02-15', '2016-04-15', 1, parallel=True, ncores=30)
