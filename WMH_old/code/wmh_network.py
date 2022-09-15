import theano
import numpy as np
from theano import tensor as T
import lasagne
from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import InputLayer, DenseLayer, batch_norm, get_output_shape, Conv2DLayer, MaxPool2DLayer, dropout, get_output
from lasagne.init import HeNormal



def generate_fusion_model(X1, X2, X3, loc_features, dropout_prob, net_arch):
	channels_num = len(net_arch["channels"])
	print("......",(channels_num, net_arch["patchSizes"][0], net_arch["patchSizes"][0]))
	
	network1 = InputLayer(shape=(None, channels_num, net_arch["patchSizes"][0], net_arch["patchSizes"][0]), input_var=X1)
	network2 = InputLayer(shape=(None, channels_num, net_arch["patchSizes"][0], net_arch["patchSizes"][0]), input_var=X2)
	network3 = InputLayer(shape=(None, channels_num, net_arch["patchSizes"][0], net_arch["patchSizes"][0]), input_var=X3)
	locTensor = InputLayer(shape=(None, 7), input_var=loc_features)
	
	#print "input shape is", (None, net_arch.channels_num, net_arch.patchSize, net_arch.patchSize)
	nr_conv_layers = len(net_arch["filterSizes"])
	nr_dense_layers = len(net_arch["denseSizes"])
	HE = HeNormal('relu')
	
	for i in range(nr_conv_layers):
		print("=============================", net_arch["filterNums"][i], (net_arch["filterSizes"][i], net_arch["filterSizes"][i]))
		network1 = batch_norm(Conv2DLayer(network1, net_arch["filterNums"][i], (net_arch["filterSizes"][i], net_arch["filterSizes"][i]), W=HE, nonlinearity=rectify))
		network1 = MaxPool2DLayer(network1, pool_size=(net_arch["poolingSizes"][i], net_arch["poolingSizes"][i]))

		network2 = batch_norm(Conv2DLayer(network2, net_arch["filterNums"][i], (net_arch["filterSizes"][i], net_arch["filterSizes"][i]), W=HE, nonlinearity=rectify))
		network2 = MaxPool2DLayer(network2, pool_size=(net_arch["poolingSizes"][i], net_arch["poolingSizes"][i]))

		network3 = batch_norm(Conv2DLayer(network3, net_arch["filterNums"][i], (net_arch["filterSizes"][i], net_arch["filterSizes"][i]), W=HE, nonlinearity=rectify))
		network3 = MaxPool2DLayer(network3, pool_size=(net_arch["poolingSizes"][i], net_arch["poolingSizes"][i]))

	network1 = batch_norm(DenseLayer(dropout(network1, p=dropout_prob), net_arch["denseSizes"][0], W=HE, nonlinearity=rectify))			
	network2 = batch_norm(DenseLayer(dropout(network2, p=dropout_prob), net_arch["denseSizes"][0], W=HE, nonlinearity=rectify))			
	network3 = batch_norm(DenseLayer(dropout(network3, p=dropout_prob), net_arch["denseSizes"][0], W=HE, nonlinearity=rectify))			

	network = lasagne.layers.ConcatLayer([network1, network2], axis=1)
	network = lasagne.layers.ConcatLayer([network, network3], axis=1)
	if net_arch["addLocationFeatures"]==1:
		network = lasagne.layers.ConcatLayer([network, locTensor], axis = 1)
	for i in range(1,nr_dense_layers):
		network = batch_norm(DenseLayer(dropout(network, p=dropout_prob), num_units = net_arch["denseSizes"][i], W=HE, nonlinearity=rectify))
	
	network = batch_norm(DenseLayer(network, num_units=2, W=HE, nonlinearity=softmax))
	
	print("Generated network...")
	return network

def save_parameters(network, fname):
	np.savez(fname, *lasagne.layers.get_all_param_values(network))

def load_parameters(network, fname):
	with np.load(fname) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)
	return network


def load_network(lpd_path, net_path):
	lp, arch, error = load_learning_process_description(lpd_path)
	ftensor4 = T.TensorType('float32', (False,)*4)
	X1 = ftensor4()
	X2 = ftensor4()
	X3 = ftensor4()
	Y = T.fmatrix()
	Loc_features = T.fmatrix()
	network = generate_fusion_model(X1, X2, X3, Loc_features, lp["dropoutProb"], arch)
	network = load_parameters(network, net_path)
	prediction = lasagne.layers.get_output(network)
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	prediction_function = theano.function(inputs=[X1,X2,X3,Loc_features], outputs=test_prediction, allow_input_downcast=True, on_unused_input='ignore')
	return prediction_function, lp, arch

def load_train_and_prediction_functions(lp, arch):
	ftensor4 = T.TensorType('float32', (False,)*4)
	X1 = ftensor4()
	X2 = ftensor4()
	X3 = ftensor4()
	Y = T.fmatrix()
	Loc_features = T.fmatrix()
	network = generate_fusion_model(X1, X2, X3, Loc_features, lp["dropoutProb"], arch)

	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(T.clip(prediction, 0.0001, 0.9999), Y)
	l2_loss =  lp["lambda2"] * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
	loss = loss.mean() + l2_loss
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.adam(loss, params, learning_rate=lp['learningRates'][0], beta1=0.9, beta2=0.999, epsilon=1e-08)
	train_function = theano.function([X1, X2, X3, Loc_features, Y], [loss, l2_loss, prediction], updates=updates, allow_input_downcast=True, on_unused_input='ignore')

	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	prediction_function = theano.function(inputs=[X1, X2, X3, Loc_features], outputs = test_prediction, allow_input_downcast=True, on_unused_input='ignore')

	return train_function, prediction_function, loss, l2_loss, prediction, params, network, X1, X2, X3, Loc_features, Y


def update_learning_rate_in_train_function(lr, loss, l2_loss, prediction, params, network, X1, X2, X3, Loc_features, Y):
	updates = lasagne.updates.adam(loss, params, learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
	train_function = theano.function([X1, X2, X3, Loc_features, Y], [loss, l2_loss, prediction], updates=updates, allow_input_downcast=True, on_unused_input='ignore') 
	return train_function




def load_learning_process_description(path):
	f = open(path)	
	keywords = ["filterSizes", "filterNums","poolingSizes","denseSizes","outClassNum","channels","patchSizes","addLocationFeatures","trainingSetPath", "validationSetPath", "maxEpochs", "learningRates", "minibatchSize", "dropoutProb","lambda1","lambda2","earlyStoppingTelorance","chunkSize","label","mask","locationFeatures", "trainingCases", "testCases", "validationCases"]
	types = ['listint', 'listint', 'listint','listint','int','liststring','listint', 'int', 'string', 'string', 'int', 'listfloat', 'int', 'float', 'float', 'float', 'int', 'int','string','string','liststring', 'string', 'string', 'string']
	found = [0]*len(keywords)
	vals = [0]*len(keywords)

	for line in f:
		#[var, val] = line.translate(None," ")[:-1].split("=")
		line = line[:-1]
		if len(line)<4:
			continue
		[var, val] = line.replace(' ', '')[:-1].split("=")
		
		print (var + "===="+val)
		for counter in range(len(keywords)):
			if var==keywords[counter]:
				vals[counter] = val
				found[counter] = 1
				break
	error = False
	for i in range(len(keywords)):
		if found[i]==0:
			print ("error value for "+keywords[i] + " not found!")
			error = True
		elif types[i]=='int':
			vals[i]=int(vals[i])
		elif types[i]=='float':
			vals[i]=float(vals[i])
		elif types[i]=='listint':
			vals[i] = [int(x) for x in (vals[i])[1:-1].split(",")]
		elif types[i]=='listfloat':
			vals[i] = [float(x) for x in (vals[i])[1:-1].split(",")]
		elif types[i]=='liststring':
			vals[i] = [x for x in (vals[i])[1:-1].split(",")]
			
	if error==False:
		print ("================================")
		for i in range(len(keywords)):
			print (keywords[i] + ":"+ str(vals[i]))
			
	net_arch = {}
	learning_params = {}
	for i in range(len(keywords)):
		if i<8:
			net_arch[keywords[i]] = vals[i]
		else:
			learning_params[keywords[i]] = vals[i]
	return [learning_params, net_arch, error]
