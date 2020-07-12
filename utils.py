
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import keras
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import pandas as pd
from pandas.plotting import scatter_matrix
import time

# avoid the dreaded type 3 fonts...
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 10})
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = True

batch_size = 64
# epochs = 2
l_rate = 0.001
num_classes = 10
img_rows, img_cols = 28, 28 # input image dimensions

col_dic={}
col_dic['rand'] = 'cyan'
col_dic['gabor'] = 'magenta'
col_dic['gabor_noise'] = 'magenta'
col_dic['exact'] = 'darkviolet'
col_dic['feats'] = 'indianred'

data_labels_dict={}
data_labels_dict['mnist']=[0,1,2,3,4,5,6,7,8,9]
data_labels_dict['fash_mnist']=['tshirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','boot']
data_labels_dict['cifar']=['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']


def get_data(data_in='mnist'):
	# mnist fash_mnist cifar10
	if data_in == 'mnist':
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		n_channels=1
		img_rows, img_cols = 28, 28
	elif data_in == 'fash_mnist':
		(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
		n_channels=1
		img_rows, img_cols = 28, 28
	elif data_in == 'cifar':
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
		n_channels=3
		img_rows, img_cols = 32, 32

	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], n_channels, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], n_channels, img_rows, img_cols)
		input_shape = (n_channels, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
		input_shape = (img_rows, img_cols, n_channels)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return x_train, y_train, x_test, y_test, input_shape


def create_model(filter_size=7, n_conv_layers=1, n_first_filters=16,
				 is_dropout=False, is_final_hidden=False, n_final_hidden=128,input_shape=(28,28,1)):
	'''  
	filter_size: first layer conv filters only
	n_conv_layers: how many convolutional layers?
	n_first_filters: how many filters in first layer
	is_dropout: whether to add on dropout
	is_final_hidden: whether to add a final hidden layer
	n_final_hidden: if is_final_hidden, how many units?
	'''

	init_in = 'he_normal' # he_normal glorot_normal # previous plots used glorot

	model = Sequential()
	model.add(Conv2D(n_first_filters, kernel_size=(filter_size, filter_size),strides=(2,2),activation='relu',kernel_initializer=init_in,
					 input_shape=input_shape))
	# model.add(Conv2D(n_first_filters, kernel_size=(filter_size, filter_size), padding='same', activation='relu',kernel_initializer=init_in))#
	# model.add(BatchNormalization())#
	model.add(MaxPooling2D(pool_size=(2, 2)))
	if is_dropout: model.add(Dropout(0.25))

	for i in range(n_conv_layers-1):
		model.add(Conv2D(n_first_filters*2**(i+1), (3, 3), activation='relu',padding='same',kernel_initializer=init_in))
		# model.add(Conv2D(n_first_filters*2**(i+1), (3, 3), activation='relu',padding='same',kernel_initializer=init_in))#
		# model.add(BatchNormalization())#
		# model.add(MaxPooling2D(pool_size=(2, 2)))#
		if is_dropout: model.add(Dropout(0.25))

	model.add(Flatten())

	# model.add(Dense(512, activation='relu',kernel_initializer=init_in))#

	if is_final_hidden:
		model.add(Dense(n_final_hidden, activation='relu',kernel_initializer=init_in))
		if is_dropout: model.add(Dropout(0.25))

	# model.add(Dense(num_classes, activation='softmax',kernel_initializer=init_in))
	
	# split this so can get logits later
	model.add(Dense(num_classes,kernel_initializer=init_in))
	model.add(keras.layers.Activation('softmax'))
	return model

def create_fc_model(n_hidden=128,n_layers=2,input_shape=(28,28,1)):
	init_in = 'he_normal' # he_normal glorot_normal # previous plots used glorot

	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	for _ in range(n_layers):
		model.add(Dense(n_hidden, activation='relu',kernel_initializer=init_in))
	# split this so can get logits later
	model.add(Dense(num_classes,kernel_initializer=init_in))
	model.add(keras.layers.Activation('softmax'))
	return model

def create_lenet(filter_size=5, n_conv_layers=1, n_first_filters=16,
				 is_dropout=False, is_final_hidden=False, n_final_hidden=128,input_shape=(28,28,1)):

	init_in = 'he_normal' # he_normal glorot_normal # previous plots used glorot
	init_in_1 = keras.initializers.RandomNormal(mean=0., stddev=0.001)

	model = keras.Sequential()
	# classic lenet (but with relus and more filters and one extra layer)
	model.add(keras.layers.Conv2D(filters=n_first_filters, kernel_size=(filter_size, filter_size), activation='relu',kernel_initializer=init_in, kernel_regularizer=keras.regularizers.l2(0.0000001),input_shape=input_shape))
	
	if n_conv_layers>1:
		model.add(keras.layers.AveragePooling2D())

		model.add(keras.layers.Conv2D(filters=n_first_filters*2, kernel_size=(filter_size, filter_size), activation='relu',kernel_initializer=init_in))
		model.add(keras.layers.AveragePooling2D())

		if n_conv_layers>2:
			model.add(keras.layers.Conv2D(filters=n_first_filters*4, kernel_size=(3, 3), activation='relu',kernel_initializer=init_in))
			model.add(keras.layers.AveragePooling2D())

	model.add(keras.layers.Flatten())

	model.add(keras.layers.Dense(units=128, activation='relu',kernel_initializer=init_in))
	model.add(keras.layers.Dense(units=84, activation='relu',kernel_initializer=init_in))

	model.add(keras.layers.Dense(num_classes,kernel_initializer=init_in))
	model.add(keras.layers.Activation('softmax'))

	return model

def filter_visualise(model_in, title_in='',norm=False,rows_in=4,filter_layer=0,n_filters=999,is_save=False):
	# allows to visualise filters
	filters = model_in.layers[filter_layer].get_weights()[0]

	n_filters = np.minimum(filters.shape[-1],n_filters)
	rows = rows_in
	cols = n_filters/rows
	fig = plt.figure(figsize=(cols/2,rows/2))


	# filters = np.clip(filters,a_min=np.percentile(filters,1),a_max=np.percentile(filters,99))
	f_min, f_max = filters.min(), filters.max()
	filters = (filters - f_min) / (f_max - f_min + 0.0001)

	for i in range(n_filters):
		# ax = plt.subplot(cols, rows, ix)
		ax = fig.add_subplot(rows, cols, i+1)
		ax.set_xticks([])
		ax.set_yticks([])

		if filters.shape[-2]==1:
			# add vmin vmax so doesn't normalise per panel
			if norm:
				# ax.imshow(filters[:, :, :, i],cmap='gray')
				ax.imshow(filters[:, :, 0, i],cmap='gray')
			else:
				# ax.imshow(filters[:, :, :, i],cmap='gray',vmin=0., vmax=1.)
				# ax.imshow(filters[:, :, 0, i],cmap='gray',vmin=0., vmax=1.)
				ax.imshow(filters[:, :, 0, i],cmap='gray',vmin=-1.5, vmax=1.5)
		else:
			# ax.imshow(filters[:, :, :, i]) # assumes RGB input
			# , not really sure what the resnet etcused
			ax.imshow(np.flip(filters[:, :, :, i],axis=-1)) # assumes BGR input
			# ax.axis('off')
			# ax.imshow(filters[:, :, :, i],vmin=-0.1, vmax=0.1)

		# ax.imshow(np.zeros_like(filters[:, :, 0, i])+i/n_filters,cmap='gray',vmin=0., vmax=1.)
		# if i == 0:
			# ax.set_title(title_in)

	fig.show()
	if is_save:
		print('filter size ',filters.shape[0], ' by ',filters.shape[1])
		fig.savefig('00_outputs/filter_'+title_in+time.strftime("%Y%m%d_%H%M%S")+'.pdf', format='pdf', dpi=500, bbox_inches='tight')
	# fig.tight_layout()
	return


def train_model(model, epochs_in, l_rate, x_train,y_train,x_test,y_test,val_size=2000):
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(l_rate),
		metrics=['accuracy'])
	hist = model.fit(x_train[0:], y_train[0:],batch_size=batch_size,epochs=epochs_in,verbose=1,
		validation_data=(x_test[0:val_size], y_test[0:val_size]))
	return hist

def entropy_calc(dist_in):
	# this returns entropy in nats
	# for x in np.arange(0,10):
	# 	x/=10
	# 	print(x)
	# 	dist_in=np.array([x,1-x])

	entropy=0.
	for i in range(dist_in.shape[0]):
		entropy -= dist_in[i]*np.log(np.maximum(dist_in[i],1e-5))
	# print(entropy)
	return entropy

def predict_classes(model,x_test,title_in='',show_plot=True,method_in='None',is_save=False):
	# run all samples through the NN and bin which class they're predicted
	# good test of prior

	y_preds = model.predict(x_test, verbose=0)
	y_preds_class = np.argmax(y_preds,axis=-1)

	# need to compute entropy of this distribution too
	# and return
	counts = np.zeros(num_classes)
	for i in range(num_classes):
		counts[i] = np.sum(y_preds_class == i)
	dist = counts/counts.sum()
	entropy=entropy_calc(dist)

	if show_plot:
		fig, ax = plt.subplots(figsize=(2.5,1.8))
		# ax.set_title(title_in)
		ax.hist(y_preds_class)
		# ax.hist(y_preds_class,color='r',label=[])
		ax.set_xlim([0,9])
		ax.set_xlabel('Predicted class')
		ax.set_ylabel('Frequency')
		bins_list = np.arange(0.,11,1) 
		ax.set_yticklabels([])
		ax.set_yticks([])
		x_in = np.arange(0,10)
		ax.set_xticks(x_in)
		if method_in in col_dic.keys():
			# print('ssfsffss')
			ax.hist(y_preds_class,color=col_dic[method_in], alpha=1, lw=0.5,edgecolor='k')
			# ax.hist(y_preds_class,alpha=0.5,label='' ,color='green',bins=bins_list,density=True)
		else:
			ax.hist(y_preds_class,color='b', alpha=1, lw=1,edgecolor='k',density=True)
		plt.text(0.05, 0.85, 'entropy$='+str(round(entropy,2))+'$', transform=ax.transAxes, fontsize=14)
		fig.show()
		if is_save:
			print('saving...')
			fig.savefig('00_outputs/egs_entropy_'+title_in+time.strftime("%Y%m%d_%H%M%S")+'.pdf', format='pdf', dpi=500, bbox_inches='tight')

	return entropy

def predict_logits(model,x_test):
	# run samples through the NN and record logits

	layer_mid = keras.Model(inputs=model.input,outputs=model.layers[-2].output)
	output_mid = layer_mid.predict(x_test)
	return output_mid

	# or use this to return probabilities after softmax
	# y_preds = model.predict(x_test, verbose=0)
	# y_preds_class = np.argmax(y_preds,axis=-1)
	# return y_preds

def predict_feats(model,x_test):
	# run samples through the NN and record hidden features before final weight layer
	layer_mid = keras.Model(inputs=model.input,outputs=model.layers[-3].output)
	output_mid = layer_mid.predict(x_test)
	return output_mid


def gabor(param_list, width=5, ThreeD=False):
	"""Gabor feature extraction."""
	# returns width x width 2d array
	[sigma, theta, Lambda, psi, gamma] = param_list

	sigma_x = sigma
	sigma_y = float(sigma) / gamma

	xmax = int((width-1)/2) # 2 -> filters of (5,5)
	ymax = xmax
	xmin = -xmax
	ymin = -ymax
	(y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

	# rotation
	x_theta = x * np.cos(theta) + y * np.sin(theta)
	y_theta = -x * np.sin(theta) + y * np.cos(theta)

	gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
	return gb

def get_gabor_params1(filter_width=5): # use these ones for paper mnist visualisation
	# generates some random params drawn from sensible distributions
	sigma =  np.random.uniform(2,10) # think to do with scale of whole thing? was 0.5,10
	theta =  np.random.uniform(0,3.14) # orientation of waves
	Lambda = np.random.uniform(3,filter_width*2) # 1/freq of waves was 1 to *1
	psi = 	 np.random.uniform(-3.14,3.14) # phase offset..?
	gamma =  np.random.uniform(0.,3) # ellipticity was 0.2 1.5
	# for color changed: Lambda from *1->*3 , sigma from 5->3
	return [sigma, theta, Lambda, psi, gamma]

def get_gabor_params(filter_width=5): # these perform better for prior checks
	# these for entropy dists
	# generates some random params drawn from sensible distributions
	sigma =  np.random.uniform(2,10)
	theta =  np.random.uniform(0,3.14)
	Lambda = np.random.uniform(1,filter_width*1) 
	psi = 	 np.random.uniform(-3.14,3.14) 
	gamma =  np.random.uniform(0.,1.5) 
	return [sigma, theta, Lambda, psi, gamma]

def get_gabor_params_3d(filter_width=5): # for training runs
	# generates some random params drawn from sensible distributions
	sigma =  np.random.uniform(2,8) # think to do with scale of whole thing? was 0.5,10
	theta =  np.random.uniform(0,3.14) # orientation of waves
	Lambda = np.random.uniform(1,filter_width*1) # 1/freq of waves 
	psi = 	 np.random.uniform(-3.14,3.14) # phase offset..?
	gamma =  np.random.uniform(0.,2) # ellipticity was 0.2 1.5
	# for color changed: Lambda from *1->*3 , sigma from 5->3
	return [sigma, theta, Lambda, psi, gamma]

def get_gabor_params_3d2(filter_width=5): # for actual paper visualisation plots
	# generates some random params drawn from sensible distributions
	sigma =  np.random.uniform(2,4) # think to do with scale of whole thing? was 0.5,10
	theta =  np.random.uniform(0,3.14) # orientation of waves
	Lambda = np.random.uniform(2,10) # 1/freq of waves 
	psi = 	 np.random.uniform(-3.14,3.14) # phase offset..?
	gamma =  np.random.uniform(0.,3) # ellipticity was 0.2 1.5
	# for color changed: Lambda from *1->*3 , sigma from 5->3
	return [sigma, theta, Lambda, psi, gamma]

def get_gabor_params_3d1(filter_width=5): # for twitter pic
	# generates some random params drawn from sensible distributions
	sigma =  np.random.uniform(2,6) # think to do with scale of whole thing? was 0.5,10
	theta =  np.random.uniform(0,3.14) # orientation of waves
	Lambda = np.random.uniform(3,12) # 1/freq of waves 
	psi = 	 np.random.uniform(-3.14,3.14) # phase offset..?
	gamma =  np.random.uniform(0.,2) # ellipticity was 0.2 1.5
	# for color changed: Lambda from *1->*3 , sigma from 5->3
	return [sigma, theta, Lambda, psi, gamma]

def get_1st_filters(model):
	curr_filters = model.layers[0].get_weights()[0]
	return curr_filters

def set_1st_filters(model, new_filters):
	params = model.get_weights()
	params[0] = new_filters
	model.set_weights(params)

	# new_params = [new_filters,model.layers[0].get_weights()[1]]
	return model

def create_1st_filters(curr_filters,method_in='rand',sigma_g=0.2):
	filter_width = curr_filters.shape[0]
	if curr_filters.shape[-2]==1: 
		is_grayscale=True
	else:
		is_grayscale=False

	if method_in=='rand':
		new_filters = np.zeros_like(curr_filters)
		# he init
		new_filters = np.random.normal(new_filters,np.sqrt(2/(filter_width**2)))
		# new_filters = np.random.normal(new_filters,0.01)
		new_filters = new_filters / new_filters.std() * np.sqrt(2/(filter_width**2))
		return new_filters
	elif method_in=='blank':
		new_filters = np.zeros_like(curr_filters)
		return new_filters
	elif method_in=='gabor' or method_in=='gabor_noise':
		new_filters = np.zeros_like(curr_filters)
		filter_width = curr_filters.shape[0]
		for i in range(curr_filters.shape[-1]):
			if is_grayscale:
				gabor_param_list = get_gabor_params(filter_width)
			else:
				gabor_param_list = get_gabor_params_3d(filter_width)
			filter_i = gabor(gabor_param_list,filter_width)
			if is_grayscale: # grayscale
				new_filters[:,:,0,i] = filter_i
			else: # RGB filters
				rand_seed=np.random.uniform()
				for c in range(new_filters.shape[-2]):
					# unichrome features
					# 1) choose darkness
					# 2) choose spectrum

					# chance of being a monochrome filter
					if rand_seed>0.5: # 0.3 for figure
						# new_filters[:,:,c,i] = np.random.uniform(0.5,0.8)*filter_i
						new_filters[:,:,c,i] = np.random.uniform(0.7,0.8)*filter_i
					else:

						# rather than having uniform, use a binary on/off per colour band
						# new_filters[:,:,c,i] = np.random.binomial(1,0.8)*filter_i
						if c==0:
							new_filters[:,:,c,i] = np.random.uniform(-0.8,0.8)*filter_i #+ np.random.uniform(-0.2,0.2)
						if c==1:
							new_filters[:,:,c,i] = np.random.uniform(-0.8,0.8)*filter_i  #+ np.random.uniform(-0.2,0.2)
						if c==2:
							new_filters[:,:,c,i] = np.random.uniform(-0.8,0.8)*filter_i #+ np.random.uniform(-0.2,0.2)
						# this orange blue combo needs: c==0: +0.5*filter_i, c==2: -0.5*filter_i
		if method_in=='gabor_noise':
			new_filters = np.random.normal(new_filters,sigma_g)

		# adjust so same as he init
		new_filters = new_filters / new_filters.std() * np.sqrt(2/(filter_width**2))
		return new_filters

	# elif method_in=='gabor_noise':
	# 	new_filters = np.zeros_like(curr_filters)
	# 	filter_width = curr_filters.shape[0]
	# 	for i in range(curr_filters.shape[-1]):
	# 		gabor_param_list = get_gabor_params(filter_width)
	# 		filter_i = gabor(gabor_param_list,filter_width)
	# 		new_filters[:,:,0,i] = filter_i
	# 	new_filters = np.random.normal(new_filters,sigma_g)
	# 	# adjust so same as he init
	# 	new_filters = new_filters / new_filters.std() * np.sqrt(2/(filter_width**2))
	# 	return new_filters
	else:
		raise Exception('method_in not specified')

def get_final_weights(model):
	# it's [-2] because we have softmax as separate layer
	curr_ws = model.layers[-2].get_weights()[0]
	return curr_ws

def set_final_weights(model, new_ws):
	params = model.get_weights()
	params[-2] = new_ws # [-1] is final bias
	model.set_weights(params)
	return model

def plot_image(x_train,idx=0):
	fig, ax = plt.subplots(figsize=(2,2))
	if x_train.shape[-1]==1:
		ax.imshow(x_train[idx,:,:,0],cmap='gray')
	else:
		ax.imshow(x_train[idx])
	ax.set_xticks([])
	ax.set_yticks([])
	fig.show()
	return

def create_noisey(x_in):
	# create a random noise scrambled version
	# make sure same mean and var as usual
	data_mean = x_in.mean()
	data_var = x_in.var()
	x_noisey = np.zeros_like(x_in)
	x_noisey = np.random.normal(loc=x_noisey,scale=0.3)
	x_noisey = x_noisey * np.sqrt(data_var/x_noisey.var())
	x_noisey += data_mean
	return x_noisey





def plot_final_layer(ws_in, data_labels):
	# scatter correlation plot
	df = pd.DataFrame(ws_in,columns=data_labels)
	axes = scatter_matrix(df, alpha=0.1, diagonal='kde')
	corr = df.corr().values
	for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
		axes[i, j].annotate("%.2f" %corr[i,j], (0.5, 0.8), xycoords='axes fraction', ha='center', va='center')
	plt.show()
	return

def plot_corr_matrix(ws_in, data_labels, title_in='',ignore_diag=False,vlims=0.4):
	# plot heat map of corr matrix
	# ignore_diag allows to not apply colours to diagnoal

	df = pd.DataFrame(ws_in,columns=data_labels)
	corr = df.corr().values

	if ignore_diag:
		diags = np.diag(corr).copy()
		np.fill_diagonal(corr,val=None)

	fig, ax = plt.subplots(figsize=(4.5,4.5))
	im = ax.imshow(corr,cmap='plasma',vmin=-vlims, vmax=vlims)

	if ignore_diag:
		for i in range(corr.shape[0]):
			corr[i,i] = diags[i]

	ax.set_xticks(np.arange(len(data_labels)))
	ax.set_yticks(np.arange(len(data_labels)))
	ax.set_xticklabels(data_labels)
	ax.set_yticklabels(data_labels)

	# rotate the tick labels and set their alignment
	plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
			 rotation_mode='anchor')

	# loop over data dimensions and create text annotations on heatmap
	for i in range(len(data_labels)):
		for j in range(len(data_labels)):
			if i==j:
				col='gray'
			else:
				col='white'
			text = ax.text(j, i, '{:.2f}'.format(round(corr[i, j],2)),
						   ha='center', va='center', color=col)

	ax.set_title(title_in)
	fig.tight_layout()
	fig.show()
	return



def plot_scatter_corr_egs(egs_logits_dict, egs_dict,class1_id,class2_id,eg1_id,eg2_id,data_labels,method_in,title_in,is_save=False):
	fig, ax = plt.subplots(figsize=(3,3))

	x_logits = egs_logits_dict[class1_id][eg1_id]
	y_logits = egs_logits_dict[class2_id][eg2_id]
	# ax.scatter(x_logits,y_logits,s=10,c='limegreen')
	ax.scatter(x_logits,y_logits,s=10,c=col_dic[method_in])
	ax.set_xlabel('logits class ' + str(data_labels[class1_id]))
	ax.set_ylabel('logits class ' + str(data_labels[class2_id]))

	arr = egs_dict[class1_id][eg1_id].squeeze()
	im = OffsetImage(arr, zoom=1.5, cmap='gray')
	im.image.axes = ax
	xy = (0.8,0.2)
	ab = AnnotationBbox(im, xy, xycoords='axes fraction',bboxprops={"edgecolor" : "none"})
	ax.add_artist(ab)

	arr = egs_dict[class2_id][eg2_id].squeeze()
	im = OffsetImage(arr, zoom=1.5, cmap='gray')
	im.image.axes = ax
	xy = (0.2,0.8)
	ab = AnnotationBbox(im, xy, xycoords='axes fraction',bboxprops={"edgecolor" : "none"})
	ax.add_artist(ab)

	corr_egs = np.corrcoef(x_logits,y_logits)[0,1]
	text = ax.text(0.05, 0.05, 'Correlation='+'{:.2f}'.format(round(corr_egs,2)), color='k',transform=ax.transAxes)

	# ax.set_title(title_in)
	fig.tight_layout()
	fig.show()
	if is_save:
		print('saving...')
		# fig.savefig('00_outputs/01_logits_corr/logitscorr_'+title_in+time.strftime("%Y%m%d_%H%M%S")+'.pdf', format='pdf', dpi=500, bbox_inches='tight')
		fig.savefig('00_outputs/01_logits_corr/logitscorr_'+title_in+'.pdf', format='pdf', dpi=500, bbox_inches='tight')

	return


def get_class_egs(x_test,y_test,n_egs):
	# get some examples of each class
	egs_dict={}
	for i in range(num_classes):
		# find 100 egs of class i
		y_test_ids = y_test.argmax(axis=-1)
		x_test_class = x_test[y_test_ids == i][:n_egs]
		egs_dict[i]=x_test_class
	return egs_dict



