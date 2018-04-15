__author__ = 'trimi'

import numpy as np
import sys
import random
from random import randint
import operator
import csv
import time
import gzip
import math
# from saveParameters import SaveParameters

class TVBPR:

    def __init__(self, iterations, K, K2, nUsers, nItems, imageFeatures, nBins,  userItems, list_of_items, nr_of_pos_events, max_timestamp, feat,itemIdToInt, min_timestamp, test_per_user, val_per_user, nr_days):
        print 'initialization started...'
        sys.setrecursionlimit(5000)
        self.nUsers = nUsers
        self.nItems = nItems + 1 + 1
        self.nr_of_pos_events = nr_of_pos_events

        self.K = K
        self.K2 = K2
	
	self.nEpochs = nBins
	self.nr_days = nr_days

        self.userItems = userItems
        self.itemIdToInt = itemIdToInt
        self.list_of_items = list_of_items

        self.nBins = nBins
        self.maxTime = max_timestamp
        self.minTime = min_timestamp
        self.min_in_seconds = min_timestamp

	n_days = int((self.maxTime - self.minTime)/86400)
	self.nDays = n_days + 1

        self.itemIdToInt = itemIdToInt
        self.feat = feat
  
        self.test_per_user = test_per_user
        self.val_per_user= val_per_user

        # biases
        self.b_u = np.zeros(self.nUsers)
        self.b_i = np.zeros(self.nItems)
	
	self.bi_bin = np.zeros((self.nItems, nBins))

        #non - visual factors
        self.gamma_user = np.random.random((self.nUsers + 1, K))
        self.gamma_item = np.random.random((self.nItems, K))
	
        # visual factors
        self.theta_user = np.random.random((self.nUsers + 1, K2))
        self.theta_item_per_bin = []
        for b in range(self.nBins):
            theta_item = np.zeros((self.nItems, K2))
            self.theta_item_per_bin.append(theta_item)

        self.E = np.random.random((K2, imageFeatures))
        self.E_t = np.random.random((nBins, K2, imageFeatures))

        # visual bias
        self.betta_cnn = np.random.random(imageFeatures)
        self.betta_cnn_t = []
        for i in range(nBins):
            b_cnn = np.zeros(imageFeatures)
            self.betta_cnn_t.append(b_cnn)

        self.J_t = np.zeros((nBins, K2)) # weighting vector for the embedding matrix
        self.C_t = np.zeros((nBins, imageFeatures)) # wieghting vector for the visual bias

        self.betta_item_visual_per_bin = np.zeros((nBins, self.nItems))

	epochs = self.init_epoch()
        self.epochs = epochs

        print 'initialization finished...'

        print 'training started...'
        self.train(iterations)
        print 'training finished...'

        print 'getting visual features...'
        self.getVisualFactors()
        print 'visual features calculation finished.'
	
	# print 'initialization of epochs started...'	
        # apply DP
        # 1. calculate pos_items_per_bin
        self.votes_per_bin = self.pos_item_per_bin()

        # 2 init DP
        self.memo, self.sol = self.dp_init()

        # DP
        # self.DP(1000)

        print 'evaluating using AUC...'

        auc = self.AUC_(self.test_per_user, self.val_per_user)

	f = open('./AUC_TVBPR.txt', 'a')
	f.write('iterations = ' + str(iterations) + '| epochs =  ' + str(nBins) + '| non_vf = ' + str(K) + '| vf = ' + str(K2))
	f.write('| AUC = ' + str(auc))
	f.close()

	# print 'AUC: ', auc 
	# print 'epochs: ', self.epochs
        print 'evaluation finished...' 
	print 'AUC: ', auc

	print 'getting visual scores...' 
	self.getScores()

    def train(self, nr_of_iterations):
        for i in range(nr_of_iterations):
            print 'iteration: ', i + 1
            start_one_iter = time.time()
            self.oneIteration()
            end_time_iter = time.time()
            time_elapsed_one_iter = end_time_iter - start_one_iter
            print 'time elapsed = ', time_elapsed_one_iter, 'sec.'
    def oneIteration(self):

        pos_per_user = {}
	pos_item_of_userID = {}
        for k,v in self.userItems.items():
            pos_per_user[k] = []
	    pos_item_of_userID[k] = []
	    for elem in self.userItems[k]:
		item_id_ = elem[0]
		pos_per_user[k].append((item_id_, elem[2]))
		pos_item_of_userID[k].append(item_id_)
               #  pos_per_user[k].append((self.userItems[k][i][0], self.userItems[k][i][2]))

	
        for i in range(self.nr_of_pos_events): #self.nr_of_pos_events
            # print 'update: ', i + 1, '/', self.nr_of_pos_events #self.nr_of_pos_events
            #sample user
            userID = self.sampleUser()
	    if  len(self.userItems[userID]) < 2:
		continue

            if len(pos_per_user[userID]) is 0:
                pos_per_user[userID] = []
                # for i in range(1, len(self.userItems[userID])):
		for elem in self.userItems[userID]:
		    pos_per_user[userID].append((elem[0], elem[2]))
                    # pos_per_user[userID].append((self.userItems[userID][i][0], self.userItems[userID][i][2]))

            """if userID == None:
                for i in range(1, len(pos_per_user)):
                    print 'user: ', i ,', len = ', len(pos_per_user[i])
                continue"""
            #sample positive item
            """if not pos_per_user[userID]:
                continue"""
            pos_event = random.choice(pos_per_user[userID])
            pos_item = pos_event[0]
           
            voteTime = pos_event[1]
            voteTime = int(voteTime)

            epoch  = self.timeInEpoch(voteTime)

            pos_per_user[userID].remove(pos_event)

            #sample negative item
            neg_item = self.sample_neg_item(pos_item_of_userID[userID])
            """if neg_item not in self.feat:
                continue"""

            #update factors
            self.updateFactors(userID, pos_item, neg_item, epoch, voteTime)

    # updating factors
    def updateFactors(self, userID, pos_item, neg_item, epoch, voteTime):

        pos_feat = self.feat[pos_item].toarray()
        neg_feat = self.feat[neg_item].toarray()
        pos_feat = pos_feat[0]
        neg_feat = neg_feat[0]

        feat_i = [] # the features of the pos_item
        feat_j = [] # the features of the neg_item
        for i in range(4096):
            feat_i.append((i, pos_feat[i]))
            feat_j.append((i, neg_feat[i]))

        diff = []
        #sparse representation\
        p_i = 0
        p_j = 0

	len_of_pos_feat = len(pos_feat)
	len_of_neg_feat = len(neg_feat)

	# start_time_f = time.time()
        while p_i < len_of_pos_feat and p_j < len_of_neg_feat:
            ind_i = feat_i[p_i][0]
            ind_j = feat_j[p_j][0]
            if ind_i < ind_j:
                diff.append((ind_i, feat_i[p_i][1]))
                p_i = p_i + 1
            elif ind_i > ind_j:
                diff.append((ind_j, feat_j[p_j][1]))
                p_j = p_j + 1
            else:
                diff.append((ind_i, feat_i[p_i][1] - feat_j[p_j][1]))
                p_i = p_i + 1
                p_j = p_j + 1
        while p_i < len(feat_i):
            diff.append(feat_i[p_i])
        while p_j < len(feat_j):
            diff.append((feat_j[p_j][0], - feat_j[p_j][1]))
	    
	# print 'epoch', epoch
 	# end_time_f = time.time()
	# print 'time for features = ', end_time_f - start_time_f
	
	# start_time_the = time.time()
        # E * (x_i - x_j)
        theta_i = np.zeros(self.K2)
        theta_i_per_bin = np.zeros(self.K2)
	len_of_diff = len(diff)
        for r in range(self.K2):
            for ind in range(len_of_diff):
                c = diff[ind][0]
		feat_val = diff[ind][1]

                theta_i[r] += self.E[r][c] * feat_val
		theta_i_per_bin[r] += self.E_t[epoch][r][c] * feat_val
	    # print theta_i[r]
	    # print theta_i_per_bin[r]
	# end_time_the = time.time()
	# print 'time theata i and theta_i_per_bin = ', end_time_the - start_time_the

        visual_score = 0
        for k in range(self.K2):
            visual_score += self.theta_user[userID][k] * (theta_i[k] * self.J_t[epoch][k] + theta_i_per_bin[k])
	
	# print 'visual score = ', visual_score

        visual_bias = 0
        for ind in range(len_of_diff):
            c = diff[ind][0]
            visual_bias += (self.betta_cnn[c] * self.C_t[epoch][c] +  self.betta_cnn_t[epoch][c]) * diff[ind][1]

	
        # x_uij = prediction(user_id, pos_item_id, bin) - prediction(user_id, neg_item_id, bin);
        pos_i = self.itemIdToInt[pos_item]
        neg_i = self.itemIdToInt[neg_item]
        
	# x_uij = (self.b_i[pos_i] + self.bi_bin[pos_i][epoch]) - (self.b_i[neg_i] + self.bi_bin[neg_i][epoch])
    	x_uij = self.b_i[pos_i] - self.b_i[neg_i]    

	# x_uij += factors_of_pos_item - factors_of_neg_item
	x_uij += np.inner(self.gamma_user[userID], self.gamma_item[pos_i]) - np.inner(self.gamma_user[userID], self.gamma_item[neg_i])
  
	x_uij += visual_score + visual_bias
	# print 'x_uij = ', x_uij
        deri = 1/(1 + np.exp(x_uij))

        self.b_i[pos_i] += 0.005 * (deri - 1 * self.b_i[pos_i])
        self.b_i[neg_i] += 0.005 * (-deri - 1 * self.b_i[neg_i])

        # updating latent factors
        # start_non_vf = time.time()
        for k in range(self.K):
            uf = self.gamma_user[userID][k]
            _if = self.gamma_item[pos_i][k]
            _jf = self.gamma_item[neg_i][k]
	    # uf_t = self.gamma_user_t[userID][k][day_ind]
	    # auk = self.alpha_uk[userID][k]

            self.gamma_user[userID][k] += 0.005 * (deri * (_if - _jf) - 1 * uf)
            self.gamma_item[pos_i][k] += 0.005 * (deri * uf - 1 * _if)
            self.gamma_item[neg_i][k] += 0.005 * (-deri * uf - 1/10.0 * _jf)
	
	# end_non_vf = time.time()
	# time_elapsed_non_vf = end_non_vf - start_non_vf
        # print 'time elapsed for non visual factors: ', time_elapsed_non_vf, ' sec.'

        # updating visual factors
        # start_vf = time.time()
        for k2 in range(self.K2):
            v_uf = self.theta_user[userID][k2]
            j_t = self.J_t[epoch][k2]

            for ind in range(len_of_diff):
                c = diff[ind][0]
		common = deri * v_uf * diff[ind][1]

                self.E[k2][c] += 0.005 * (common * j_t)
                self.E_t[epoch][k2][c] += 0.005 * (common - 0.0001 * self.E_t[epoch][k2][c])
		

            self.theta_user[userID][k2] += 0.005 * (deri * (theta_i[k2] * j_t + theta_i_per_bin[k2]) - 1 * v_uf)
            self.J_t[epoch][k2] += 0.005 * (deri * theta_i[k2] * v_uf - 0.0001 * j_t)
	
	# updating visual bias 
        for ind in range(len_of_diff):
            c = diff[ind][0]
            c_tf = self.C_t[epoch][c]
            b_cnn = self.betta_cnn[c]
	    
	    common_2 = 0.005 * deri * diff[ind][1]
 
            self.betta_cnn[c] += common_2 * c_tf
            self.C_t[epoch][c] += common_2 * b_cnn - 0.005 * 0.0001 * c_tf
            self.betta_cnn_t[epoch][c] += common_2 - 0.005 * 0.0001 * self.betta_cnn_t[epoch][c]
        # end_vf = time.time()
        # time_elapsed_vf = end_vf - start_vf 
        # print 'time elapsed on visual factors: ', time_elapsed_vf,' sec.'
    # sample user
    def sampleUser(self):
        userId_ = randint(1, self.nUsers)

        return userId_

    #sample negative item
    def sample_neg_item(self, pos_per_user):
        while True:
            neg_item = random.choice(self.list_of_items)
            if neg_item not in pos_per_user:
                return neg_item



    # calculate the corresponding epoch given timestamp
    def timeInEpoch(self, timesta):
        inte = (self.maxTime - self.minTime)/80
        # print 'interval = ', inte
        b_ind = np.minimum(79, int((timesta - self.minTime)/inte))
        # print 'b_ind =', b_ind
        for i in range(len(self.epochs)):
            start_bin = self.epochs[i][0]
            end_bin = self.epochs[i][1]
            if end_bin >= b_ind:
		# print 'epochs ==', self.epochs
		# print 'epoch_ind = ', i
                return i

    # calculate pos_item_per_bin
    def pos_item_per_bin(self):
        votes_per_bin = []
        for i in range(80):
            votes_per_bin.append([])

        # adding the samples from the validation set to the corresponding bin
        for k,v in self.val_per_user.items():
            user = k
            item = self.val_per_user[k][0]
            rating = self.val_per_user[k][1]
            voteTime = self.val_per_user[k][2]
            interv = (self.maxTime - self.minTime)/80
            # print 'interval = ', inte
            b_ind = np.minimum(79, int((voteTime - self.minTime)/interv))
            votes_per_bin[b_ind].append((user, item))

        return votes_per_bin

    # epoch initialization
    def init_epoch(self):
        epochs = []
        interval = 80/self.nEpochs
        bin_from = 0
        bin_to = interval - 1
        for i in range(self.nEpochs):
            if epochs == []:
                epochs.append((bin_from, bin_to))
            else:
            	bin_from = bin_to + 1
            	bin_to = bin_to + interval
            	epochs.append((bin_from, bin_to))

        return epochs

    # DP
    def DP(self, num_of_neg_items):
        pos_per_user = self.pos_per_user_()
        sampleMap = {}
        for k,v in self.userItems.items():
            user = k
            for i in range(num_of_neg_items):
                neg_item = self.sample_neg_item(pos_per_user[user])
                if user not in sampleMap:
                    sampleMap[user] = [neg_item]
                else:
                    sampleMap[user].append(neg_item)

        # apply DP
        fval = self.f(0, 79, 0, 10, sampleMap)
        print 'f() = ', fval

        new_epochs = []
        start = 0
        end = 79
        ep = 0
        pieces = 10
        last_bin_to = -1

        for i in range(10):
            seprator = self.sol[start][end][ep][pieces - 1]

            if seprator == -1:
                print 'Exception: No solution found by DP'

            if new_epochs == []:
                binFrom = last_bin_to + 1
                binTo = last_bin_to = seprator
                new_epochs.append((binFrom, binTo))
            else:
                binFrom = last_bin_to + 1
                binTo = last_bin_to = seprator
                new_epochs.append((binFrom, binTo))

            start = seprator + 1
            pieces = pieces - 1
            ep += 1
	    
  	# print 'epochs = ', self.epochs
        self.epochs = new_epochs
  	# print 'new_epochs = ', self.epochs
       

    # positive items per user
    def pos_per_user_ (self):
        pos_per_user = {}
        for k,v in self.userItems.items():
            for i in range(len(self.userItems[k])):
                item__ = self.userItems[k][i][0]
                if k not in pos_per_user:
                    pos_per_user[k] = [item__]
                else:
                    pos_per_user[k].append(item__)

        return pos_per_user

    # f()
    def f(self, startBin, endBin, epo, pieces, sample_map):
        if self.memo[startBin][endBin][epo][pieces - 1] != sys.float_info.max:
            return self.memo[startBin][endBin][epo][pieces - 1]

        if pieces == 1:
            self.memo[startBin][endBin][epo][0] = self.onePieceVal(startBin, endBin, epo, sample_map)
            if self.memo[startBin][endBin][epo][0] < sys.float_info.max:
                self.sol[startBin][endBin][epo][0] = endBin
            else:
                self.sol[startBin][endBin][epo][0] = -1
            return self.memo[startBin][endBin][epo][0]

        max_val = -sys.float_info.max
        for k in range(startBin, endBin - pieces):
            val = self.f(startBin, k, epo, 1, sample_map) + self.f(k + 1, endBin, epo + 1, pieces - 1, sample_map)
            if val > max_val:
                max_val = val
                self.sol[startBin][endBin][epo][pieces - 1] = k
        self.memo[startBin][endBin][epo][pieces - 1] = max_val
        return max_val

    def onePieceVal(self,startBin, endBin, epo, sampleMap):
        res = 0
        total = 0
        for i in range(startBin, endBin + 1):
            if self.memo[i][i][epo][0] != sys.float_info.max:
                res += self.memo[i][i][epo][0]
                continue
            for j in range(len(self.votes_per_bin[i])):
                user_ = self.votes_per_bin[i][j][0]
                item_ = self.votes_per_bin[i][j][1]
		item_id_int = self.itemIdToInt[item_]
                x_ui = self.prediction(user_, item_id_int, epo)

                for k in range(len(sampleMap[user_])):
                    neg_item = sampleMap[user_][k]
	   	    neg_item_ = self.itemIdToInt[neg_item]
                    x_uj = self.prediction(user_, neg_item_, epo)

                    total += np.log(self.sigmoid(x_ui - x_uj))

            self.memo[i][i][epo][0] = total
            self.sol[i][i][epo][0] = i
            res += total

        return res

    # sigmoid function
    def sigmoid(self, x):
        sig = 1 / (1 + np.exp(-x))
        return sig

    # DP init
    def dp_init(self):
        memo = np.zeros((80,80,10,10))
        sol = np.zeros((80,80,10,10))
        for b in range(80):
            for j in range(80):
                for k in range(10):
                    for x in range(10):
                        memo[b][j][k][x] = sys.float_info.max # not calculated
                        sol[b][j][k][x] = -1 # not calculated
        return memo, sol
    
    # function getting the scores for each action occured over time (on daily basis), such as: user-items interaction score, item bias score, etc.
    def getScores(self):
	self.nr_days = self.nr_days + 1
	visualBiasDays = [[] for _ in range(self.nr_days)]
	nonVisualBiasDays = [[] for _ in range(self.nr_days )]
	totalBias = [[] for _ in range(self.nr_days)]
	visualInteractionDays = [[] for _ in range(self.nr_days)]
	nonVisualInteractionDays = [[] for _ in range(self.nr_days)]
	totalInteraction = [[] for _ in range(self.nr_days)]

	for k,v in self.userItems.items():
		userID = k 
		sz = len(self.userItems[k])
		for j in range(sz): 
		

			item = self.userItems[k][j][0]
			itemID = self.itemIdToInt[item]
			rating = self.userItems[k][j][1]
			tmstamp = self.userItems[k][j][2]
			day_ind = int((tmstamp - self.minTime)/86400)
			epoch_ind = self.timeInEpoch(tmstamp)
			
			interaction = np.dot(self.gamma_user[userID], self.gamma_item[itemID]) + np.dot(self.theta_user[userID], self.theta_item_per_bin[epoch_ind][itemID]) 
			visual_interaction =  np.dot(self.theta_user[userID], self.theta_item_per_bin[epoch_ind][itemID])
			visual_bias = self.betta_item_visual_per_bin[epoch_ind][itemID]
			non_visual_bias = self.b_i[itemID]
			total_bias = non_visual_bias + visual_bias

			totalInteraction[day_ind].append(interaction)
			visualInteractionDays[day_ind].append(visual_interaction)
			visualBiasDays[day_ind].append(visual_bias)
			nonVisualBiasDays[day_ind].append(non_visual_bias)
			totalBias[day_ind].append(total_bias)
	print 'writing the scores...'
	f = open('./scores_total_interactions_TVBPR.txt','w')
	f.write(str(totalInteraction))
	f.close()
	f = open('./scores_visual_interactions_TVBPR.txt','w')
        f.write(str(visualInteractionDays))
        f.close()
	f = open('./scores_visual_bias_TVBPR.txt','w')
        f.write(str(visualBiasDays))
        f.close()
	f = open('./scores_non_visual_bias_TVBPR.txt','w')
        f.write(str(nonVisualBiasDays))
        f.close()
	f = open('./scores_total_bias_TVBPR.txt','w')
        f.write(str(totalBias))
        f.close()
	print 'writting finished!'
	
    # gettin the visual factors
    def getVisualFactors(self):
        for bin in range(self.nEpochs):
	    print '-------------EPOCH------------ ', bin 
            visual_scores = []
	    dim = [[] for _ in range(self.K2)]

            for i in range(len(self.list_of_items)):
                asin = self.list_of_items[i]
                itemID = self.itemIdToInt[asin]
               
                item_feat = self.feat[asin].toarray()
                item_feat = item_feat[0]

                for k in range(self.K2):
                    for f in range(4096):
                        self.theta_item_per_bin[bin][itemID][k] += (self.E[k][f] * self.J_t[bin][k] + self.E_t[bin][k][f]) * item_feat[f]

                # visual bias
                for f in range(4096):
                    self.betta_item_visual_per_bin[bin][itemID] += (self.betta_cnn[f] * self.C_t[bin][f] + self.betta_cnn_t[bin][f]) * item_feat[f]
		
         	for k in range(self.K2):
			dim[k].append((asin, self.theta_item_per_bin[bin][itemID][k]))
	
		# calculating the visual score of the item
                vp = 0
                for u in range(1, self.nUsers + 1):
                    vp += np.inner(self.theta_user[u], self.theta_item_per_bin[bin][itemID])

                vp = vp/self.nUsers
                visual_scores.append((asin, vp))


            max_vs = max(visual_scores, key=operator.itemgetter(1))


            print 'itemID with max. visual score: ', max_vs[0],' visualScore = ', max_vs[1]

		
	    f = open('./top_items_TVBPR.txt', 'a')
	    # printing top items for given epoch and visual dimension

            for d in range(self.K2):
		max_value = max(dim[d], key=operator.itemgetter(1))
		dim[d].sort(key=operator.itemgetter(1), reverse=True)
		top1 = dim[d][0]
		top2 = dim[d][1]
		top3 = dim[d][2]
		top4 = dim[d][3]
		top5 = dim[d][4]

                # print 'dim:',d
		# print '1st item: ', top1, ', 2nd item: ', top2, ', 3d item: ', top3, ', 4th item: ', top4, ', 5th item: ', top5

		
		f.write('\ndim: ' + str(d) + '\n top items: ' + str(top1) + ',' + str(top2) + ',' + str(top3) + ',' + str(top4) + ',' + str(top5))

	    f.close()

    # function for calculating the prediction value
    def prediction(self, userID, itemID, epoch_):

  	pred = self.b_i[itemID] + np.dot(self.gamma_user[userID], self.gamma_item[itemID]) +  np.dot(self.theta_user[userID], self.theta_item_per_bin[epoch_][itemID]) + self.betta_item_visual_per_bin[epoch_][itemID]
        
	return pred
	
    # function for evaluating the model, i.e. AUC applied
    def AUC_ (self, test_per_us, val_per_us):
	countusers = 0
        AUC_  = np.zeros(self.nUsers + 1)
        
        for k,v in test_per_us.items():
            user_ = k
	 
            # print 'USER: ', user_, ' being tested...'
            test_item_asin = test_per_us[k][0]
	    if k in val_per_us:
                val_item_asin = val_per_us[k][0]
		
            if test_item_asin not in self.feat:
                continue
            # print test_item_asin

            item_intID = self.itemIdToInt[test_item_asin]

            time_d = int(test_per_us[k][2])
	    #if time_d < 1374105600:
 		# continue
	
	    countusers += 1
	    day_ind_ = int((time_d - self.minTime)/86400)
            # print test_per_us[k]
            # print time_d
            # bin_ind = self.calBin(time_d)

            epoch = self.timeInEpoch(time_d)
            # print time_d
            # print 'bin ind = ', bin_ind
          
            pred_of_test = self.prediction(user_, item_intID, epoch)


            asins_of_user = []
            for a in range(len(self.userItems[user_])):
                asins_of_user.append(self.userItems[user_][a][0])

            count = 0
            count_val = 0
            maxx  = 0.0
            for i in range(len(self.list_of_items)):
                asin = self.list_of_items[i]
                item_id = self.itemIdToInt[asin]
                if asin not in self.feat:
                    continue
                if asin in asins_of_user or asin == test_item_asin or asin == val_item_asin:
                    continue
                else:
                    maxx += 1
                    pred_of_neg = self.prediction(user_, item_id, epoch)

                    if pred_of_test > pred_of_neg:
                        count += 1

            AUC_[user_] = 1 * (count/maxx)

            # print 'count = ', count
            # print 'max =', maxx
             # print 'AUC for userID: ', user_,' is: ', AUC_[user_]

        auc = 0
        num_users = len(test_per_us)
        # print 'users = ', num_users

        for i in range(len(AUC_)):
            auc += AUC_[i]

        # print 'AUC = ', auc/num_users
        # print 'bins = ', bin_indices
        # print 'auc = ', auc, ', num_users = ', countusers
	# print 'DP applied!'

        return auc/countusers









