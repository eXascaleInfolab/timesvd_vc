import numpy as np
import sys
import random
import operator
import csv
import time
import gzip
import math
import struct

from scipy.sparse import csr_matrix


class LoadData:

	def __init__(self, dataset):

		self.dataset = dataset
		if dataset == '390_actions':
			self.stop_point = 389
		elif dataset == '780_actions':
			self.stop_point = 779
		elif dataset == '1560_actions':
			self.stop_point = 1559
		elif dataset == '2340_actions':
			self.stop_point = 2339
		elif dataset == '4099_actions':
			self.stop_point = 4099
		

	# load rating file
	def readRatings(self, path):
		with open(path, "rb") as f:
		    
		    useridToInt_ = {}
		    intToUserid = {}
		    asinToInt_ = {}
		    userItems_ = {}
		    itemUsers_ = {}
		    timestamps_ = []
		    itemsList = []
		    userCount  = 1
		    itemCount = 1

		    count = 0

		    while True:
		        fa = f.readline()
		        a = fa.split(',')
		
		        user_id = a[0]
		        item_asin = a[1]
		        rating__ = float(a[2])
		        timestamp_ = int(a[3])

		        # user ID to int
		        if user_id not in useridToInt_:
		            useridToInt_[user_id] = userCount
		            userCount += 1

		        user_int =  useridToInt_[user_id]

		        # ASIN to int
		        if item_asin not in asinToInt_:
		            asinToInt_[item_asin] = itemCount
		            itemCount += 1
			
			# lsit of ASIN items
			if item_asin not in itemsList:
		            itemsList.append(item_asin)

		        #user-items dictionary
			if user_int not in userItems_:
			    userItems_[user_int] = [(item_asin, rating__, timestamp_)]
			else:
			    if item_asin not in userItems_[user_int]:
				userItems_[user_int].append((item_asin, rating__, timestamp_))

		        # item-users dictionary
		        if item_asin not in itemUsers_:
		            itemUsers_[item_asin] = [(user_int, rating__, timestamp_)]
		        else:
		            if user_int not in itemUsers_[item_asin]:
		                itemUsers_[item_asin].append((user_int, rating__, timestamp_))

		        print count
			
			# if count == 4099: # for the dataset with 78 users and > 4000 actions			
			# if count == 2339: # for the dataset with 78 users and 2340 actions			
			# if count == 1559: # for the dataset with 78 users and 1560 actions
			# if count == 779: # for the dataset with 78 users and 780 actions
			if count == self.stop_point: # dataset: 78 users and 390 actions
		            break
		        if count % 1000 == 0:
		            print count
		        count = count + 1

		return  userItems_, itemsList

	# loading image features
	def readImFeat(self, path):
		    asinToInt = {}
		    intToAsin = {}
		    b = 0
		    with open(path, "rb") as f:
		        while True:
		            asin = f.read(10)
		            if asin == '': break
		            asinToInt[asin] = b
		            intToAsin[b] = asin
		            b = b + 1
		            feat = []
		            for i in range(4096):
		                feat.append(struct.unpack('f', f.read(4)))
		            yield asin, feat, asinToInt, intToAsin

    	# loading image features
	def loadImgFeat(self, items):
		
		# filtering out the items for which there is no features available
		imageFeat = self.readImFeat("./image_features/image_features_Men.b")
		
		itemFeatures = {}
		total_nr_of_items = len(items)
		ma = 58.388599
		e = 0
		id = 1
		count_items_in_asin = 0
		asinToInt = {}
		intToAsin = {}
		while True:
		        if e == 369053:
		            break

		        feat = []
		        v = imageFeat.next()
		        asin = v[0]

		        features = v[1]
		        e = e + 1


		        if asin in items:
		            count_items_in_asin = count_items_in_asin + 1
		            # print "count_items_in_asin = ", count_items_in_asin
			    if count_items_in_asin % 50 == 0:
				print 'Currently, items found having image features for: ', count_items_in_asin, '/', total_nr_of_items
		            # print '#item: ', e
		            # print asin
		            asinToInt[asin] = id
		            intToAsin[id] = asin
		            c = []
		            for f in range(4096):
		                c.append(features[f][0]/ma)
		            feat = csr_matrix(c)
		            itemFeatures[asin] = feat
		            id = id + 1
		        else:
		            continue

		return itemFeatures, asinToInt

	#filtering the data given the available list of the items containing the image features
	def filter_user_items(self, userItems, imageFeatures):

		new_user_items = {} # new user_items by filtering the items for which there are no image features available
		new_items_list = [] # list of all items that have been rated and the image features exist for
		timestamps = []
		# filtering - only items that image features have been extracted for are considered
		for k,v in userItems.items():
	  		sz = len(userItems[k])

	    		for i in range(sz):
				item = userItems[k][i][0]
				timestamp = userItems[k][i][2]

				if item not in imageFeatures: 
			    		continue

				if k not in new_user_items:
			    		new_user_items[k] = []

				new_user_items[k].append(userItems[k][i])

        			if item not in new_items_list:
            				new_items_list.append(item)

				timestamps.append(timestamp)
			
			min_timestamp = min(timestamps)
			max_timestamp = max(timestamps)

    		return new_user_items, new_items_list, min_timestamp, max_timestamp, timestamps

	# creating the testing set
	def create_testing_set(self, userItems):
		test_per_user = {}
		for k,v in userItems.items():
		   if k not in test_per_user:
			test_per_user[k] = userItems[k][0]
			del userItems[k][0]
		return test_per_user, userItems

	# creating the validation set
	def create_validation_set(self, userItems):
		val_per_user = {}
		for k,v in userItems.items():
		    if k not in val_per_user:	
			val_per_user[k] = userItems[k][1]
			del userItems[k][1]
		return val_per_user, userItems

	def get_nr_of_pos_events(self, userItems):
		pos_events = 0
		for k, v in userItems.items():
			pos_events += len(userItems[k])
		return pos_events

	def run_load_data(self, path):
		# read the rating file
		userItems, itemsAsinList = self.readRatings(path)
		
		# load the image features
		print 'Loading image features and filtering...'
		itemImageFeatures, asinToInt = self.loadImgFeat(itemsAsinList)
		print 'Loading image features finished.'
		
		# new user-items dictionary with the filtered list of items
		new_userItems, new_itemsAsinList, min_timestamp, max_timestamp, timestamps = self.filter_user_items(userItems, itemImageFeatures)

		# create the test set
		test_per_user, new_userItems = self.create_testing_set(new_userItems)

		# create the validation set
		val_per_user, new_userItems = self.create_validation_set(new_userItems)
		
		# get the positive events with regard to the training set
		pos_events = self.get_nr_of_pos_events(new_userItems)
		

		itemAsinToInt = {}
		intId = 0
		for j in range(len(new_itemsAsinList)):
		     if new_itemsAsinList[j] not in itemAsinToInt:
			itemAsinToInt[new_itemsAsinList[j]] = intId
			intId += 1
		nr_days = (max_timestamp - min_timestamp)/86400 # 86400 seconds correspond to 1 day

		return new_userItems, new_itemsAsinList, itemImageFeatures, itemAsinToInt, test_per_user, val_per_user, min_timestamp, max_timestamp, nr_days, pos_events, timestamps














