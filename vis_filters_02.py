import importlib
import utils
importlib.reload(utils)
from utils import *

# avoid the dreaded type 3 fonts...
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 10})
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = True

K.clear_session()

# this script displays first layer filters of pretrained popular CNNs
# the lenent needs to be trained from scratch, others are downloaded through keras

model_str = 'vgg' # resnet vgg lenet
is_save = False
is_norm = True # whether to normalise cmap of filters per filter

if model_str=='resnet':
	# 7 x 7 filter size
	from keras.applications.resnet50 import ResNet50
	model_ = ResNet50
	filter_layer = 2
elif model_str=='vgg':
	# 3 x 3 filter size
	from keras.applications.vgg16 import VGG16
	model_ = VGG16
	filter_layer = 1

if model_str in ['resnet','vgg']:
	# visualise trained filters	
	model = model_(weights='imagenet')
	print(model.summary())
	filter_visualise(model,norm=is_norm,title_in=model_str+'learnt',filter_layer=filter_layer,is_save=is_save,n_filters=32)

	# random filters
	model = model_(weights=None)
	filter_visualise(model,norm=is_norm,title_in=model_str+'rand',filter_layer=filter_layer,is_save=is_save,n_filters=32)



if model_str=='lenet':
	# no pretrained versions for this, so we do it ourselves...
	dataset_str = 'mnist' # mnist fash_mnist
	filter_size=5
	filter_layer=0
	# l_rate=0.01 # 5 eps of this then 5 eps 0.001
	# l_rate=0.001

	x_train, y_train, x_test, y_test, input_shape = get_data(dataset_str)

	model = create_lenet(filter_size=filter_size,n_conv_layers=2,n_first_filters=32,input_shape=input_shape)
	print(model.summary())

	filter_visualise(model,title_in=model_str+'rand',norm=is_norm,filter_layer=filter_layer,is_save=is_save,n_filters=32)

	train_model(model,l_rate=0.01,epochs_in=2,val_size=1000,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
	train_model(model,l_rate=0.001,epochs_in=4,val_size=1000,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
	# train_model(model,l_rate=0.0001,epochs_in=2,val_size=1000,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
	filter_visualise(model,title_in=model_str+'learnt',norm=is_norm,filter_layer=filter_layer,is_save=is_save,n_filters=32)
	# filter_visualise(model,title_in=model_str+'learnt',norm=True,filter_layer=filter_layer,is_save=False,n_filters=32)



