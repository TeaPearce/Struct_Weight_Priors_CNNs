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

# this script runs experiments for section 4.4, testing training curves

is_save = False
is_norm = True # whether to normalise cmap of filters per filter
filter_size = 5
# sigma_g=0.2 # 0.3 for grayscale
# filter_layer=0
# is_colour=False
# dataset_str = 'mnist' # fash_mnist mnist
# n_conv_layers=1
n_runs=1 # 5


n_egs=20 # how many egs to use for feats method of last layer


# for dataset_str, n_conv_layers, sigma_g in [('mnist',1,0.),('mnist',1,0.2),('fash_mnist',1,0.),('fash_mnist',1,0.2),('mnist',3,0.),('fash_mnist',3,0.)]:

# for dataset_str, n_conv_layers, sigma_g in [('mnist',1,0.),('mnist',2,0.),('mnist',3,0.)]:
# for dataset_str, n_conv_layers, sigma_g in [('fash_mnist',1,0.),('fash_mnist',2,0.),('fash_mnist',3,0.)]:
# for dataset_str, n_conv_layers, sigma_g in [('cifar',1,0.),('cifar',2,0.),('cifar',3,0.)]:
for dataset_str, n_conv_layers, sigma_g in [('fash_mnist',3,0.)]:
	# methods = ['rand','gabor_feats'] # these specify what type of prior is activated, looks for if 'gabor' or 'feats' is in the string
	methods = ['rand']
	# methods = ['rand','feats']
# for dataset_str, n_conv_layers, sigma_g, methods in [('mnist',1,0.,['rand','featsonly']),('mnist',1,0.,['rand','gaboronly']),('mnist',1,0.,['rand','gabor_feats']),('mnist',1,0.2,['rand','gaboronly'])]:

	print('\n\n -- new set ',dataset_str, n_conv_layers, sigma_g)
	x_train, y_train, x_test, y_test, input_shape = get_data(dataset_str) # mnist fash_mnist
	
	if dataset_str=='mnist':
		n_epochs=10 # 10
	elif dataset_str=='fash_mnist':
		n_epochs=15
	elif dataset_str=='cifar':
		n_epochs=15 # 15

	if dataset_str=='cifar':
		input_shape=(32,32,3)
	else:
		input_shape=(28,28,1)
	data_labels = data_labels_dict[dataset_str]

	results={}
	for method_loop in methods:
		results[method_loop]=[]
		for run in range(n_runs):
			print('\nmethod=',method_loop,' run=',run)

			if 'gabor' in method_loop:
				is_gabor=True
			else:
				is_gabor=False

			if 'feats' in method_loop:
				is_feats=True
			else:
				is_feats=False
			
			# 1) init first half of NN
			K.clear_session()
			if 'fc' in method_loop:
				model = create_fc_model(n_hidden=1024,n_layers=n_conv_layers,input_shape=input_shape)
			else:
				# model = create_model(filter_size=filter_size,n_conv_layers=n_conv_layers,n_first_filters=16)
				model = create_model(filter_size=filter_size,n_conv_layers=n_conv_layers,n_first_filters=16,input_shape=input_shape)

				curr_filters = get_1st_filters(model)
				if is_gabor:
					new_filters = create_1st_filters(curr_filters, method_in='gabor_noise',sigma_g=sigma_g)
				else:
					new_filters = create_1st_filters(curr_filters, method_in='rand',sigma_g=sigma_g)
				model = set_1st_filters(model, new_filters)
				# filter_visualise(model)

			# 2) pass a few examples of each class through the NN and record activation features in final layer
			# need to randomise which examples grabbing
			start_idx = np.random.randint(0,int(x_test.shape[0]-(n_egs*num_classes*2)))
			egs_dict = get_class_egs(x_test[start_idx:],y_test[start_idx:],n_egs)

			egs_feats_dict={}
			for i in range(num_classes):
				y_feats = predict_feats(model,egs_dict[i])
				egs_feats_dict[i] = y_feats.T

			# 3) use the mean of the activations per class as the mean of a Gaussian to draw weights from
			ws_init = get_final_weights(model)
			new_ws = np.zeros_like(ws_init)
			# if method_loop == 'rand':
			if not is_feats:
				cov_mat = np.eye(new_ws.shape[-1])
				new_ws = np.random.multivariate_normal(new_ws[0],cov=cov_mat,size=new_ws.shape[0])
			# elif method_loop == 'feats':
			elif is_feats:
				n_feats = egs_feats_dict[0].shape[0]
				# for f in range(10):
				for f in range(n_feats):
					feat_means=[]; feat_stds=[]
					for i in range(num_classes):
						feat_means.append(egs_feats_dict[i][f].mean())
						feat_stds.append(egs_feats_dict[i][f].std()+1e-3)
					
					feat_means -= np.mean(feat_means)

					if True:
						# using this slight adjustment instead of means directly
						# gives a v. slight improvement...
						# I call this 'binary feats'

						# if feat_means <0, set to -1, else +1
						feat_means[feat_means<0]=-1
						feat_means[feat_means>=0]=1

					# new_ws[f,:] = np.random.normal(loc=feat_means,scale=feat_stds/np.sqrt(n_egs))
					new_ws[f,:] = np.random.normal(loc=feat_means,scale=feat_stds)
					# new_ws[f,:] = np.random.normal(loc=feat_means,scale=0.1)

					# new_ws[f,:] = np.random.normal(loc=feat_means,scale=1e-9)
			
			# plot_final_layer(new_ws, data_labels)


			# plot_corr_matrix(new_ws, data_labels, title_in='pre normalise '+method_loop,ignore_diag=True)

			# 4) renormalise to he init
			new_ws -= new_ws.mean()
			new_ws = new_ws/new_ws.std() * np.sqrt(2/(ws_init.shape[0]))
			set_final_weights(model, new_ws)

			# plot_corr_matrix(new_ws, data_labels, title_in='pre train '+method_loop,ignore_diag=True)

			if False:
				model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(l_rate),
								metrics=['accuracy'])
				metrics = model.evaluate(x_test,y_test)
				print('pre training accuracy:',metrics[1])

				# this allows us to test how well we're doing on the data we've used for the prior
				for i in range(num_classes):
					y_egs = np.zeros_like(y_test[0:egs_dict[0].shape[0]])
					y_egs[:,i]=1   
					metrics = model.evaluate(egs_dict[i],y_egs,verbose=0)
					print('acc on train dist class',i,' is ',round(metrics[1],2))

			# hist = model.fit(x_train[0:n_egs*num_classes], y_train[0:n_egs*num_classes],batch_size=batch_size,epochs=10,verbose=1,
			# 	validation_data=(x_test[0:1000], y_test[0:1000]))

			# results[method_loop].append(hist)

			# metrics = model.evaluate(x_test,y_test)
			# print('after training accuracy:',metrics[1])

			# record training curves
			results[method_loop].append(train_model(model,l_rate=l_rate,epochs_in=n_epochs,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,val_size=x_test.shape[0]))
			# ws_final = get_final_weights(model)
			# plot_corr_matrix(ws_final, data_labels, title_in='after train '+method_loop,ignore_diag=True)
			# filter_visualise(model)

	fig, ax = plt.subplots(figsize=(4,3))
	for method_loop in methods:
		mean_accs=[]
		mean_vals=[]
		label=''
		if 'fc' in method_loop:
			col_in='red'
			label+='FC '
		if 'gabor' in method_loop:
			col_in='magenta'
			label+='Gabor '
		if 'feats' in method_loop:
			col_in='magenta'
			label+='Feats '
		if label=='':
			col_in='cyan'
			label+='i.i.d.'
		label_train=label+' train'
		label_test=label+' test'
		for i in range(n_runs):
			# if i==n_runs-1: 
			# label=''
			# if 'gabor' in method_loop:
			# 	col_in='magenta'
			# 	label+='Gabor '
			# if 'feats' in method_loop:
			# 	col_in='magenta'
			# 	label+='Feats '
			# if label=='':
			# 	col_in='cyan'
			# 	label+='i.i.d.'
			# label_train=label+' train'
			# label_test=label+' test'
			# else: 
				# label_train=None
				# label_test=None
			ep_list = range(1,len(results[method_loop][i].history['acc'])+1)
			# ax.plot(ep_list,results[method_loop][i].history['acc'],c=col_in,alpha=0.6,lw=0.5,label=label_train)
			# ax.plot(ep_list,results[method_loop][i].history['acc'],c=col_in,alpha=0.5,lw=0.5)
			# ax.plot(ep_list,results[method_loop][i].history['val_acc'],c=col_in,alpha=0.6,lw=0.5,ls='--',label=label_test)
			# ax.plot(ep_list,results[method_loop][i].history['val_acc'],c=col_in,alpha=0.5,lw=0.5,ls='--')
			mean_accs.append(results[method_loop][i].history['acc'])
			mean_vals.append(results[method_loop][i].history['val_acc'])
		# now plot mean
		mean_accs=np.array(mean_accs)
		mean_vals=np.array(mean_vals)
		ax.plot(ep_list,mean_accs.mean(axis=0),c=col_in,alpha=1,lw=1,label=label_train)
		ax.plot(ep_list,mean_vals.mean(axis=0),c=col_in,alpha=1,lw=1,ls='--',label=label_test)

	fig.legend(loc=(0.45,0.2))
	ax.set_xlabel('Epochs')
	ax.set_ylabel('Accuracy')
	# ax.set_title('training runs '+dataset_str+' fsize='+str(filter_size)+' convs='+str(n_conv_layers))
	fig.show()
	if is_save:
		print('saving...')
		fig.savefig('00_outputs/02_trainings/trainings_'+dataset_str+'_'+method_loop+'_layers'+str(n_conv_layers)+'_sigg'+str(round(sigma_g,2)).replace('.','')+'_'+time.strftime("%Y%m%d_%H%M%S")+'.pdf', format='pdf', dpi=500, bbox_inches='tight')



