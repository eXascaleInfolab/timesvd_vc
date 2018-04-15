import numpy as np
import sys
import random
import operator
import csv
import time
import gzip
import math

from loading_data import LoadData

from timeSVDpp import timeSVDpp
from TVBPR import TVBPR
from timeSVD_VC import timeSVD_VC

if len(sys.argv) != 7:
	  print "Parameters: "
	  print "1. model name (e.g. timeSVDpp/ TVBPR/ timeSVD_VC)"
	  print "2. number of iterations"
	  print "3. number of epochs/bins"
	  print "4. number of non-visual factors"
	  print "5. number of visual factors"
	  print "6. dataset (e.g. 390_actions, 780_actions, 1560_actions, 2340_actions, 4099_actions)"
	  sys.exit()


model = sys.argv[1]
nIterations = int(sys.argv[2])
nEpochs = int(sys.argv[3])
nFactors = int(sys.argv[4])
nVisualFactors = int(sys.argv[5])
dataset = sys.argv[6]

if dataset == '390_actions':
	path = "./datasets/trainingSet_78_users_390_actions.csv" 
elif dataset == '780_actions':
	path = "./datasets/trainingSet_78_users_780_actions.csv" 
elif dataset == '1560_actions':
	path = "./datasets/trainingSet_78_users_1560_actions.csv" 
elif dataset == '2340_actions':
	path = "./datasets/trainingSet_78_users_2340_actions.csv" 
elif dataset == '4099_actions':
	path = "./datasets/trainingSet_78_users_4099_actions.csv" 
else: 
	print 'The dataset does not exist!' 


# running timeSVD++
if model == 'timeSVDpp':
	#loading the data
	ld = LoadData(dataset)
	userItems, itemsAsinList, itemImageFeatures, itemAsinToInt, test_per_user, val_per_user, min_timestamp, max_timestamp, nr_days, posEvents, timestamps = ld.run_load_data(path)


	nUsers = len(userItems) 
	nItems = len(itemsAsinList)
	nBins = nEpochs

	#running the model
	timesvdpp = timeSVDpp(nIterations, nFactors, nUsers, nItems, nBins, userItems, itemsAsinList, itemAsinToInt, min_timestamp, max_timestamp,  test_per_user, val_per_user, timestamps)

# running TVBPR
elif model == 'TVBPR':
	# loading the data
	ld = LoadData(dataset)
	userItems, itemsAsinList, itemImageFeatures, itemAsinToInt, test_per_user, val_per_user, min_timestamp, max_timestamp, nr_days, posEvents, timestamps = ld.run_load_data(path)

	nUsers = len(userItems) 
	nItems = len(itemsAsinList)
	nBins = nEpochs
	nImageFeatures = 4096

	# running the model
	tvbpr = TVBPR(nIterations, nFactors, nVisualFactors, nUsers, nItems, nImageFeatures, nEpochs, userItems, itemsAsinList, posEvents, max_timestamp, itemImageFeatures, itemAsinToInt, min_timestamp, test_per_user, val_per_user, nr_days)

# running timeSVD_VC
elif model == "timeSVD_VC":
	# loading the data
	ld = LoadData(dataset)
	userItems, itemsAsinList, itemImageFeatures, itemAsinToInt, test_per_user, val_per_user, min_timestamp, max_timestamp, nr_days, posEvents, timestamps = ld.run_load_data(path)

	nUsers = len(userItems) 
	nItems = len(itemsAsinList)
	nBins = nEpochs
	nImageFeatures = 4096
	
	# running the model
	timesvd_vc = timeSVD_VC(nIterations, nFactors, nVisualFactors, nUsers, nItems, nImageFeatures, nEpochs, userItems, itemsAsinList, posEvents, max_timestamp, itemImageFeatures, itemAsinToInt, min_timestamp, test_per_user, val_per_user, nr_days)
		

else:
	print 'Choose the right model!'


