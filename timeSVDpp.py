__author__ = 'trimi'

import numpy as np
import math
import csv
import operator
import time 

class timeSVDpp:
    def __init__(self, iter_, nFactors, nUsers, nItems, nBins, userItems, list_of_all_items, item_asin_to_int, minTimeInSec, maxTimeInSec, test_per_user, val_per_user, timestamps):

        self.gamma_1 = 0.005
        self.gamma_2 = 0.007
        self.gamma_3 = 0.001
        self.g_alpha = 0.00001
        self.tau_6 = 0.005
        self.tau_7 = 0.015
        self.tau_8 = 0.015
        self.l_alpha = 0.0004

        self.minTimeInSec = minTimeInSec
        self.maxTimeInSec = maxTimeInSec

        # number of days
        nDays = int((self.maxTimeInSec - self.minTimeInSec) / 86400)
	self.nBins = nBins
        self.timestamps = timestamps
        self.max_time = nDays
        # self.min_time = 1896
        self.min_time = 0
        # self.min_time_in_seconds = min_time_in_seconds
        self.min_time_in_seconds = 1000
        # self.userDays = userDays

        self.iterations = iter_
        self.userItems = userItems
        # self.userIdToInt = userIDToInt
        self.test_per_user = test_per_user
        self.val_per_user = val_per_user

        self.itemAsinToInt = item_asin_to_int
        self.factors = nFactors + 1
        self.nUsers = nUsers + 1
        self.nItems = nItems + 1
        self.nBins = nBins
	self.K = nFactors
        #self.nDays = 214
        # for #users = 30, 1963 days
        self.nDays = nDays + 1

        self.list_of_all_items = list_of_all_items
		
	self.item_bias = []
	
        #initialization
        print('initialization started...')
        b_u, b_i, u_f, i_f, y_j, sumMW, bi_bin, alpha_u, bu_t, alpha_u_k, userFactors_t, cu, cu_t = self.init(self.nUsers, self.nItems, self.factors, self.nBins)

        self.bu = b_u
        self.bi = b_i
        self.bi_bin = bi_bin
        self.alpha_u = alpha_u
        self.bu_t = bu_t

        self.cu = cu
        self.cu_t = cu_t

        self.userFactors = u_f
        self.itemFactors = i_f
        self.y_j = y_j
        self.sumMW = sumMW

        self.alpha_u_k = alpha_u_k
        self.userFactors_t = userFactors_t
        print 'initialization finished...'

        self.average = self.avg()
        print 'avg = ', self.average

        print 'training started...'
        self.train(self.iterations)
        print 'training finished...'

        print 'evaluation started...'
        # rmse = self.RMSE_librec()
        # print 'RMSE = ', rmse

        auc = self.AUC_(self.test_per_user, self.val_per_user)
     
	f = open('./AUC_timeSVDpp.txt', 'a')
        f.write('iterations = ' + str(self.iterations) + '| epochs =  ' + str(nBins) + '| non_vf = ' + str(nFactors))
        f.write('| AUC = ' + str(auc))
        f.close()
        print 'evaluation finished'
	print 'AUC = ', auc
        # test
	# print 'item bias'
	# print self.item_bias
	# self.getMostFav()
	self.getScores()

    def init(self, nUsers, nItems, nFactors, nBins):
        #biases
        bu = np.zeros(nUsers + 1)
        bi = np.zeros(nItems + 1) # dtype = 'float64'

        # bi_bin = np.random.random((nItems + 1, nBins))
        bi_bin = []
        for b in range(nItems + 1):
            bii = np.zeros(nBins)
            bi_bin.append(bii)

        alpha_u = np.zeros(nUsers + 1) # nUsers + 1, dtype = 'float64'

        bu_t = np.zeros((nUsers + 1, self.nDays), dtype = 'float64')
        """for us in range(1, len(bu_t)):
            if self.userDays[us] > 0:
                for g in range(len(self.userDays)):
                    day_ = self.userDays[us][g]
                    bu_t[us][day_] = np.random.uniform(0, 1)"""

        cu = np.zeros(nUsers + 1)
        cu_t = []
        for b in range(nItems + 1):
            cuu = np.zeros(self.nDays)
            cu_t.append(cuu)

        #factors
        # userFactors = np.random.random((nUsers + 1, nFactors))
        userFactors = []
        for b in range(nUsers + 1):
            bii = np.random.uniform(0, 1, nFactors)
            userFactors.append(bii)

        # itemFactors = np.random.random((nItems + 1, nFactors))
        itemFactors = []
        for b in range(nItems + 1):
            bii = np.random.uniform(0, 1, nFactors)
            itemFactors.append(bii)

        # y_j = np.random.random((nItems + 1, nFactors), np.float64)
        y_j = []
        for b in range(nItems + 1):
            bii = np.zeros(nFactors)
            y_j.append(bii)

        # sumMW = np.random.random((nUsers + 1, nFactors))
        sumMW = []
        for b in range(nUsers + 1):
            bii = np.random.random(nFactors)
            sumMW.append(bii)

        #time-based parameters
        # alpha_u_k = np.random.random((nUsers + 1, nFactors))
        alpha_u_k = []
        for b in range(nUsers + 1):
            bii = np.zeros(nFactors)
            alpha_u_k.append(bii)

        userFactors_t = np.zeros((nUsers + 1, nFactors, self.nDays))
        """for userr in range(1, len(userFactors_t)):
            if self.userDays[userr] > 0:
                for da in range(len(self.userDays)):
                    day_ = self.userDays[userr][da]
                    for fct in range(nFactors):
                        userFactors_t[userr][fct][day_] = np.random.uniform(0, 1)"""

        return bu, bi, userFactors, itemFactors, y_j, sumMW, bi_bin, alpha_u, bu_t, alpha_u_k, userFactors_t, cu, cu_t

    def train(self, iter):
        for i in range(iter):
            loss = 0
            print 'iteration: ', i + 1
	    # auc = self.AUC_(self.test_per_user, self.val_per_user)
            # print 'AUC = ', auc
	    starting_time = time.time()	
            self.oneIteration()
	    ending_time = time.time()
	    time_elapsed = ending_time - starting_time
	    print 'time_elapse = ', time_elapsed
		
    def oneIteration(self):
        loss = 0
        for userId, v in self.userItems.items():
            # print 'userID = ', userId
            if userId % 100 == 0:
                print 'users trained: ', userId
            tmpSum = np.zeros(self.factors, dtype='float')

            # if userId not in self.training_userItems:
             #   continue
            sz = len(self.userItems[userId])

            if sz > 0:

                for it in range(len(self.userItems[userId])):
                    itemid = self.userItems[userId][it][0]
                    rating = self.userItems[userId][it][1]
                    timestamp_ = self.userItems[userId][it][2]

                    item_id_to_int = self.itemAsinToInt[itemid]
                    # li = [userId, itemid, rating, timestamp_]
                    # if li not in self.training_mat:
                    #     continue

                    sqrtNum = 1/(math.sqrt(sz))

                    day_ind = int((timestamp_ - self.minTimeInSec)/86400)
                    cu_ = self.cu[userId]
                    cut_ = self.cu_t[userId][day_ind]
                    bi_ = self.bi[item_id_to_int]
                    bit_ =  self.bi_bin[item_id_to_int][self.calBin(day_ind)]
                    bu_ = self.bu[userId]
                    but_ = self.bu_t[userId][day_ind]
                    au = self.alpha_u[userId]
                    dev_ = self.dev(userId, day_ind)

		    # print 'dev = ', dev_
		    # print 'but_ = ', but_
		    # print 'au =', au 
                    sy = []
                    for a in range(self.factors):
                        res = 0
                        for it in range(sz):
                            item_id_ = self.userItems[userId][it][0]
                            int_id = self.itemAsinToInt[item_id_]
                            res += self.y_j[int_id][a]
                        res = sqrtNum * res
                        sy.append(res)


                    pred = self.average + (bi_ + bit_) #  * (cu_ + cut_)
                    pred += bu_ + au * dev_ + but_


                    for f in range(self.factors):
                        qik = self.itemFactors[item_id_to_int][f]
                        pukt = self.userFactors[userId][f] + self.alpha_u_k[userId][f] * self.dev(userId, day_ind) + self.userFactors_t[userId][f][day_ind]
                        pred += qik * (pukt + sy[f])

                    prediction = self.prediction(userId, item_id_to_int, day_ind)
                    error = pred - rating
                    loss += error * error
                
                    # user bias
                    sgd = error +  1 * bu_
                    self.bu[userId] += -0.005 * sgd
                    loss += 0.005 * bu_ * bu_

                    # item bias
                    sgd = error  + 1 * bi_
                    self.bi[item_id_to_int] += -0.005 * sgd
                    loss += 0.005 * bi_ * bi_

                    # item bias bi, bin(t)
                    sgd = error + 1 * bit_
                    self.bi_bin[item_id_to_int][self.calBin(day_ind)] += -0.005 * sgd
                    loss += 0.005 * bit_ + bit_
                    # cu
                    """sgd = error * (bi_ + bit_) + 0.01 * cu_
                    self.cu[userId] += -0.01 * sgd

                    #cu_t
                    sgd = error * (bi_ + bit_) + 0.01 * cut_
                    self.cu_t[userId][day_ind] += -0.01 * sgd"""

                    # bu_t
                    sgd = error + 1 * but_
                    delta = but_ - 0.005 * sgd
                    self.bu_t[userId][day_ind] = delta
                    loss += 0.005 * but_ + but_

                    # au
                    sgd = error * dev_ + 1 * au
                    self.alpha_u[userId] += -0.005 * sgd
                    loss += 0.005 * au + au

                    # print bi_, bit_, bu_, but_, au, dev_

                    # updating factors
                    for k in range(self.factors):
                        u_f = self.userFactors[userId][k]
                        i_f = self.itemFactors[item_id_to_int][k]
                        u_f_t = self.userFactors_t[userId][k][day_ind]
                        auk = self.alpha_u_k[userId][k]


			# print u_f, auk, dev_, u_f_t
                        pukt = u_f + auk * dev_ + u_f_t

                        # print u_f, auk, dev_, u_f_t
                        # print pukt

                        sum_yk = 0
                        for j in range(sz):
                            pos_item = self.userItems[userId][j][0]
                            pos_item_int = self.itemAsinToInt[pos_item]
                            sum_yk += self.y_j[pos_item_int][k]

                        # print u_f, i_f, u_f_t, auk, sum_yk

                        sgd = error * (pukt + sqrtNum * sum_yk) + 1 * i_f
                        self.itemFactors[item_id_to_int][k] += -0.005 * sgd
                        loss += 0.005 * i_f * i_f

                        #update user factor
                        sgd = error * i_f + 1 * u_f
                        self.userFactors[userId][k] += -0.005 * sgd
                        loss += 0.005 * u_f * u_f

                        # auk
                        sgd  = error * i_f * dev_ + 1 * auk
                        self.alpha_u_k[userId][k] += -0.005 * sgd
                        loss += 0.005 * auk * auk

                        # uf_t
                        sgd = error * i_f + 1 * u_f_t
                        delta = u_f_t - 0.005 * sgd
                        self.userFactors_t[userId][k][day_ind] = delta
                        loss += 0.005 * u_f_t * u_f_t

                        for j in range(sz):
                            itID = self.userItems[userId][j][0]
                            it_int_id = self.itemAsinToInt[itID]
                            yjk_ = self.y_j[it_int_id][k]
                            sgd = error * sqrtNum * i_f + 1 * yjk_
                            self.y_j[it_int_id][k] += -0.005 * sgd
                            loss += 0.005 * yjk_ * yjk_
        loss *= 0.5
        # print 'loss = ', loss


    #overall rating avarage
    def avg(self):
        s = 0
        count = 0

        for i, v in self.userItems.items():
            #if i not in self.training_userItems:
             #   continue
            sz = len(self.userItems[i])
            for j in range(sz):
                rating_ = self.userItems[i][j][1]
                s += rating_
                count += 1
        avg = s/count

        return avg

    #find the index of the bin for the given timestamp
    def calBin(self, day_of_rating):
        interval = (self.max_time - 0.0) / self.nBins
        bin_ind = min(self.nBins - 1, int((day_of_rating - self.min_time)/interval))

        return bin_ind

    #deviation of user u at given t
    def dev(self, userID, t):
        deviation = np.sign(t - self.meanTime(userID)) * pow(abs(t - self.meanTime(userID)), 0.2)

        return deviation

    #mean rating time for given user
    def meanTime(self, userID):
        s = 0
        count = 0
        sz = len(self.userItems[userID])
        if sz > 0:
            list_of_days = []
            for i in range(sz):
                timestamp_st = self.userItems[userID][i][2]
                d_ind = int((timestamp_st - self.minTimeInSec)/86400)
                if d_ind not in list_of_days:
                    list_of_days.append(d_ind)
                    s += d_ind
                    count += 1

            return s/count
        else:
            summ = 0
            l_of_days = []
            cc = 0
            for i in range(len(self.timestamps)):
                dind = int((self.timestamps[i] - self.minTimeInSec)/86400)
                if dind not in l_of_days:
                    l_of_days.append(dind)
                    summ += dind
                    cc += 1
            globalMean = summ/cc

            return globalMean

    def getMostFav(self):
        for bin in range(self.nBins):
	    print '------------bins------------ ', bin 
            scores = []
            dim = [[] for _ in range(self.K)]
	    vscore_list = []
            for i in range(len(self.list_of_all_items)):
                asin = self.list_of_all_items[i]
                item_id = self.list_of_all_items[i]
		item_id_to_int = self.itemAsinToInt[item_id]
		vs = 0
		for k, v in self.userItems.items():
			days_ratings = []
			sz = len(self.userItems[k])
			user_id = k
			for j in range(len(self.userItems[k])):
				timesta = self.userItems[k][j][2]
				dayInd = int((timesta - self.minTimeInSec)/ 86400)
				days_ratings.append(dayInd)
			b = 0
			for dd in range(len(days_ratings)):
				bin_ = self.calBin(days_ratings[dd])
				if bin_ == bin:
					b = bin_
					day_ind = days_ratings[dd]
					break
			sqrtNum = 1/(math.sqrt(sz))
			sy = []
                        for a in range(self.factors):
                        	res = 0
                                for it in range(sz):
                                	item_id_ = self.userItems[user_id][it][0]
                              		int_itemID = self.itemAsinToInt[item_id_]
                                 	res += self.y_j[int_itemID][a]
                                res = sqrtNum * res
                                sy.append(res)

			if b != 0:

                        # dot product between user features and item features
			
                        	user_factors = 0
                        	for f in range(self.factors):
                                	qik = self.itemFactors[item_id_to_int][f]
                                	pukt = self.userFactors[user_id][f] + self.alpha_u_k[user_id][f] * self.dev(user_id, day_ind) + self.userFactors_t[user_id][f][day_ind]
                                	user_factors += qik * (pukt + sy[f])
			else:
				user_factors = 0
                                for f in range(self.factors):
                                        qik = self.itemFactors[item_id_to_int][f]
                                        pukt = self.userFactors[user_id][f] + self.userFactors_t[user_id][f]
                                        user_factors += qik * (pukt + sy[f])

			vs += user_factors
	
		vscore = vs/len(self.userItems)
		vscore_list.append((asin, vscore))	
	max_vs = max(vscore_list, key=operator.itemgetter(1))

        # print 'itemID with max. visual score: ', max_vs[0],' visualScore = ', max_vs[1]
    def getScores(self):

	visualBiasDays = [[] for _ in range(self.nDays)]
        nonVisualBiasDays = [[] for _ in range(self.nDays)]
        totalBias = [[] for _ in range(self.nDays)]
        visualInteractionDays = [[] for _ in range(self.nDays)]
        nonVisualInteractionDays = [[] for _ in range(self.nDays)]
        totalInteraction = [[] for _ in range(self.nDays)]

        for k,v in self.userItems.items():
        	user_id = k

		sz = len(self.userItems[k])
		sqrtNum = 1/(math.sqrt(sz))

		for j in range(sz):
		
			item = self.userItems[k][j][0]
			item_id = self.itemAsinToInt[item]
			rating = self.userItems[k][j][1]
			tmstamp = self.userItems[k][j][2]
			day_ind = int((tmstamp - self.minTimeInSec)/ 86400)

			item_bias = (self.bi[item_id] + self.bi_bin[item_id][self.calBin(day_ind)])
			sy = []
        		for a in range(self.factors):
            			res = 0
            			for it in range(sz):
                			item_id_ = self.userItems[user_id][it][0]
                			int_itemID = self.itemAsinToInt[item_id_]
                			res += self.y_j[int_itemID][a]
            				res = sqrtNum * res
            			sy.append(res)

			# dot product between user features and item features
			user_factors = 0
        		for f in range(self.factors):
            			qik = self.itemFactors[item_id][f]
            			pukt = self.userFactors[user_id][f] + self.alpha_u_k[user_id][f] * self.dev(user_id, day_ind) + self.userFactors_t[user_id][f][day_ind]
            			user_factors += qik * (pukt + sy[f])

                	non_visual_interaction = user_factors
                	non_visual_item_bias = self.bi[item_id] + self.bi_bin[item_id][self.calBin(day_ind)]
 
                	# adding the values to the corresponding day
                	totalInteraction[day_ind].append(non_visual_interaction)      
                	nonVisualBiasDays[day_ind].append(non_visual_item_bias)
             

	print 'writing...'
        f = open('./scores_total_interactions_timeSVDpp.txt','w')
        f.write(str(totalInteraction))
        f.close()
 
        f = open('./scores_non_visual_bias_timeSVDpp.txt','w')
        f.write(str(nonVisualBiasDays))
        f.close()
     
    #prediction method
    def prediction(self, user_id, item_id, day_ind):

        if user_id in self.userItems:
            sz = len(self.userItems[user_id])
            sqrtNum = 1/(math.sqrt(sz))
        else:
            sz = len(self.userItems[user_id])
            sqrtNum = 1/(math.sqrt(sz))
            print 'user not trained...'

        # global mean
        prediction = self.average

        # item bias
        prediction += (self.bi[item_id] + self.bi_bin[item_id][self.calBin(day_ind)]) #  * (self.cu[user_id] + self.cu_t[user_id][day_ind])

        # user bias

        prediction += self.bu[user_id] + self.alpha_u[user_id] * self.dev(user_id, day_ind) + self.bu_t[user_id][day_ind]


        # sum of the features from the Ru set
        sy = []
        for a in range(self.factors):
            res = 0
            for it in range(sz):
                item_id_ = self.userItems[user_id][it][0]
                int_itemID = self.itemAsinToInt[item_id_]
                res += self.y_j[int_itemID][a]
            res = sqrtNum * res
            sy.append(res)

        # dot product between user features and item features
        for f in range(self.factors):
            qik = self.itemFactors[item_id][f]
            pukt = self.userFactors[user_id][f] + self.alpha_u_k[user_id][f] * self.dev(user_id, day_ind) + self.userFactors_t[user_id][f][day_ind]
            prediction += qik * (pukt + sy[f])


        return prediction


    def RMSE_librec(self):
        mean_squared_error = 0
        c = 0

        for i in range(len(self.testing_mat)):
            row = self.testing_mat[i]
            userid = row[0]
            itemid = row[1]
            rating = float(row[2])
            t_stamp = int(row[3])
            counting = 0
            day = int((t_stamp - self.min_time_in_seconds)/ 86400000)
            if userid not in self.training_userItems:
                counting += 1
                continue

            predict = self.prediction(userid, itemid, day)

            mean_squared_error += math.pow((rating - predict), 2)

            c += 1

        meanSuaredError = mean_squared_error/c
        meanSuaredError = math.sqrt(meanSuaredError)
        print 'counting: ', counting
        return meanSuaredError

    def AUC_ (self, test_per_us, val_per_us):

        AUC_  = np.zeros(self.nUsers + 1)
        for k,v in test_per_us.items():
            user_ = k
            # print 'USER: ', user_, ' being tested...'
            test_item_asin = test_per_us[k][0]
            val_item_asin = val_per_us[k][0]

            item_intID = self.itemAsinToInt[test_item_asin]

            time_d = int(test_per_us[k][2])
            # print test_per_us[k]
            # print time_d
            # bin_ind = self.calBin(time_d)


            dayInd = int((time_d - self.minTimeInSec)/ 86400)
            # print time_d
            # print 'bin ind = ', bin_ind
           
	    
            pred_of_test = self.prediction(user_, item_intID, dayInd)


            asins_of_user = []
            for a in range(len(self.userItems[user_])):
                asins_of_user.append(self.userItems[user_][a][0])

            count = 0
            count_val = 0
            maxx  = 0.0
            for i in range(len(self.list_of_all_items)):
                asin = self.list_of_all_items[i]
                item_id = self.list_of_all_items[i]
		itemIntID = self.itemAsinToInt[item_id]

                if asin in asins_of_user or asin == test_item_asin or asin == val_item_asin:
                    continue
                else:
                    maxx += 1
                    pred_of_neg = self.prediction(user_, itemIntID, dayInd)
		    """if user_ == 16 and pred_of_neg < pred_of_test:
		    	print 'test_item = ', test_item_asin, ', pred = ', pred_of_test, ', bias = ', self.bi[item_intID]
		    	print 'neg_item = ', asin, ', pred = ', pred_of_neg, ', bias_neg = ', self.bi[itemIntID]"""
		   
                    if pred_of_test > pred_of_neg:
                        count += 1

            AUC_[user_] = 1 * (count/maxx)
	    # print 'user: ', user_, ', pred_of_test = ', pred_of_test

            # print 'count = ', count
            # print 'max =', maxx
             # print 'AUC for userID: ', user_,' is: ', AUC_[user_]

        auc = 0
        num_users = len(test_per_us)
        # print 'users = ', num_users
	print 'AUC = ', AUC_ 
        for i in range(len(AUC_)):
            auc += AUC_[i]
	
        # print 'AUC = ', auc/num_users
        # print 'bins = ', bin_indices
        print 'auc = ', auc, ', num_users = ', num_users
	
	# user_items = []
	# for k,v in self.userItems.items():
	    # print 'user: ', k, ', ', self.userFactors[k]
	# print user_items
        return auc/num_users


# timeSVDplusspluss = timeSVDpp(100, 20, nUsers, nItems, nBins, user_items, items_, userIDToInt, itemAsinToInt, min(times_), max(times_),  test_per_user, val_per_user, times_)

