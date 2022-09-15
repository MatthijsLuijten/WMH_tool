import time
import sys
import theano
import math
import numpy as np
from sklearn.metrics import roc_curve, auc
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import InputLayer, DenseLayer, batch_norm, get_output_shape, Conv2DLayer, MaxPool2DLayer, dropout, get_output
from lasagne.init import HeNormal
import h5py
import wmh_utility
import wmh_network

# Mohsen Ghafoorian, 2017
# Lasagne implementation of a multi-scale Fully Convolutional Network Training

def append_file(path, line):
	file = open(path, "a")
	file.write(line+"\n")
	file.close()

	
# Get estimates of a chunked dataset
def compute_estimates(X, loc_features, prediction_function, lp, arch):
	nr_samples = X.shape[0]
	nr_chunks = int(math.ceil(float(nr_samples)/(lp["chunkSize"])))
	nr_channels = len(arch["channels"])
	chunk_size = lp["chunkSize"]
	valid_estimates = []
	asghar = 0
	for i in range(nr_chunks-1):	
		X1 = X[i*chunk_size:(i+1)*chunk_size, 0:nr_channels, :, :]
		X2 = X[i*chunk_size:(i+1)*chunk_size, nr_channels:2*nr_channels, :, :]
		X3 = X[i*chunk_size:(i+1)*chunk_size, 2*nr_channels:3*nr_channels, :, :]
		chunk_loc = loc_features[i*chunk_size:(i+1)*chunk_size,:] 
		for start in range(0,chunk_size-lp["minibatchSize"], lp["minibatchSize"]):
			end = start+lp["minibatchSize"]
			valid_estimates += prediction_function(X1[start:end, :, :, :],X2[start:end, :, :, :],X3[start:end, :, :, :], chunk_loc[start:end,:]).tolist()
		#last minibatch
		if chunk_size%lp["minibatchSize"]!=0: 
			valid_estimates += prediction_function(X1[end:, :, :, :],X2[end:, :, :, :],X3[end:, :, :, :],chunk_loc[end:,:]).tolist()
	return valid_estimates


# Compute accuracy and AUC
def evaluate_estimates(estimates, y):
	accuracy = np.mean(np.argmax(y, axis=1) == np.argmax(estimates, axis=1))
	fpr, tpr, thresholds = roc_curve(y[:, 1], estimates[:, 1])
	best_acc_val=-99999
	best_acc_thresh=0
	for i in range(len(fpr)):
		acc = (tpr[i]-fpr[i]+1.0)/2.0
		if acc>=best_acc_val:
			best_acc_val = acc
			best_acc_thresh=thresholds[i]
	roc_auc = auc(fpr, tpr)
	return accuracy, best_acc_val, best_acc_thresh, roc_auc


	
def load_data(lp):
	input_hdf5 = h5py.File(lp["trainingSetPath"], 'r')
	loc_features_file = open(lp["trainingSetPath"]+"_loc.csv")
	val_loc_features_file = open(lp["validationSetPath"]+"_loc.csv")
	train_X = input_hdf5['patches']
	train_y = input_hdf5['y']
	input_hdf5 = h5py.File(lp["validationSetPath"], 'r')
	val_X = input_hdf5['patches']
	val_y = input_hdf5['y']
	print("Data opened successfully...")
	print("Train features: ", train_X.shape)
	print("Train labels: ", train_y.shape)
	print("Validation features shape: ", val_X.shape)
	print("Validation labels shape: ", val_y.shape)
	return train_X, train_y, val_X, val_y, loc_features_file, val_loc_features_file

	

def main_loop(lp, arch, number, nets_path):
	epsilon = 0.0001
	current_net_out_path = nets_path+number+"/"+number
	best_net_out_path = nets_path+number+"/"+number+"_best.npz"
	costs_file_path = current_net_out_path+"cost.txt"
	best_val_AUC = best_val_index = 0
	report_path = current_net_out_path+"report.txt"

	train_function, prediction_function, loss, l2_loss, prediction, params, network, X1, X2, X3, loc_features, Y = wmh_network.load_train_and_prediction_functions(lp, arch)

	wmh_network.save_parameters(network, current_net_out_path + '_initialization'  + ".npz")
	
	train_X, train_y, val_X, val_y, loc_features_file, val_loc_features_file = load_data(lp)
	
	print(">>>>>training, validation set sizes", train_X.shape[0], val_X.shape[0])
	nr_samples = train_X.shape[0]
	nr_samples_val = val_X.shape[0]
	print("data loaded successfully")
	nr_chunks = int(math.ceil(float(nr_samples)/(lp["chunkSize"])))
	print("Number of  to process per epoch: ", nr_chunks, train_X.shape)

	

	print("loading location info...")
	batch_loc = np.zeros((nr_samples, len(lp["locationFeatures"])), dtype="float32")
	for i in range(nr_samples):
		batch_loc[i,:] = [float(xx) for xx in loc_features_file.readline()[:-1].split(",")]
	loc_features_file.close()
	batch_loc_valid = np.zeros((nr_samples_val, len(lp["locationFeatures"])), dtype="float32")
	for i in range(nr_samples_val):
		batch_loc_valid[i,:] = [float(xx) for xx in val_loc_features_file.readline()[:-1].split(",")]
	val_loc_features_file.close()
	
	nr_channels = len(arch["channels"])
	for epoch in range(lp["maxEpochs"]):
		pb = wmh_utility.Progbar(target=nr_chunks, width=100)
		lr = lp["learningRates"][-1]
		if epoch<len(lp["learningRates"]):
			lr = lp["learningRates"][epoch]
		start_time = time.clock()
		train_estimates = []	
		
		for i in range(nr_chunks-1):
			cost_sum=cost_l2_sum=cc=0.0
			chunk_s1 = train_X[i*lp["chunkSize"]:(i+1)*lp["chunkSize"], 0:nr_channels, :, :]
			chunk_s2 = train_X[i*lp["chunkSize"]:(i+1)*lp["chunkSize"], nr_channels:2*nr_channels, :, :]
			chunk_s3 = train_X[i*lp["chunkSize"]:(i+1)*lp["chunkSize"], 2*nr_channels:3*nr_channels, :, :]
			chunk_loc = batch_loc[i*lp["chunkSize"]:(i+1)*lp["chunkSize"],:] 
			chunk_y = train_y[i*lp["chunkSize"]:(i+1)*lp["chunkSize"], :]
			for start in range(0,lp["chunkSize"]-lp["minibatchSize"], lp["minibatchSize"]):
				end = start+lp["minibatchSize"]
				cost_temp, cost_l2_temp, pred = train_function(chunk_s1[start:end,:,:,:], chunk_s2[start:end,:,:,:], chunk_s3[start:end,:,:,:], chunk_loc[start:end,:], chunk_y[start:end,:])
				cost_sum += cost_temp
				cost_l2_sum += cost_l2_temp
				cc += 1
				train_estimates += pred.tolist()
			#last minibatch
			if lp["chunkSize"]%lp["minibatchSize"]!=0:
				cost_temp, cost_l2_temp, pred = train_function(chunk_s1[end:], chunk_s2[end:], chunk_s3[end:], chunk_loc[end:], chunk_y[end:])
				train_estimates += pred.tolist()
			
			append_file(costs_file_path, str(cost_sum/cc)+","+str(cost_l2_sum/cc))
			pb.update(current=i+1)

		### Done with this epoch. Get train and validation errors ###
		#progress_bar.end()
		epoch_time = (time.clock() - start_time)/60
		print("Time this epoch: {t} minutes \n".format(t=epoch_time))
		append_file(costs_file_path, "=====================================================================")

		print("Computing training estimates...")
		train_accuracy, train_acc_best, train_best_thresh, train_AUC = evaluate_estimates(np.asarray(train_estimates), train_y[:len(train_estimates)])
		report_str = "training(acc, accBest,bestThresh,auc): ("+ str(train_accuracy)+ ", "+str(train_acc_best)+ ", "+ str(train_best_thresh)+ ", "+str(train_AUC)+")"
		print(report_str)
		append_file(report_path, report_str)
		print("computing validation acc and auc...")
		val_estimates = compute_estimates(val_X, batch_loc_valid, prediction_function, lp, arch)
		np_val_estimates = np.asarray(val_estimates)
		val_accuracy, val_acc_best, val_best_thresh, val_AUC = evaluate_estimates(np_val_estimates, val_y[:np_val_estimates.shape[0]])
		report_str = "validation(acc, accBest,bestThresh,auc): ("+str(val_accuracy)+ ", "+ str(val_acc_best)+ ","+ str(val_best_thresh)+ ","+ str(val_AUC)+")"
		print(report_str)
		append_file(report_path, report_str)
		append_file(report_path, "-----------------------------------------")
		wmh_network.save_parameters(network, current_net_out_path + '_' + str(epoch) + ".npz")
		
		if val_AUC >= best_val_AUC+epsilon:
			best_val_index = epoch
		if val_AUC > best_val_AUC:
			wmh_network.save_parameters(network, best_net_out_path)
			best_val_AUC = val_AUC        
		if best_val_index+lp["earlyStoppingTelorance"]<=epoch:
			print("Early stopping activated!")
			break
		
		lr = lp["learningRates"][-1]
		if epoch<len(lp["learningRates"]):
			lr = lp["learningRates"][epoch]
		train_function = wmh_network.update_learning_rate_in_train_function(lr, loss, l2_loss, prediction, params, network, X1,X2,X3,loc_features,Y)
		

	def run(nets_path, number):
		params_path = nets_path+number+"/"+number+".lpd"
		learning_params, net_arch, error = wmh_network.load_learning_process_description(params_path)
		if error==False: 
			main_loop(learning_params, net_arch, number, nets_path)
