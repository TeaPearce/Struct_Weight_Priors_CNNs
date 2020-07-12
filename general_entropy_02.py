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

# this script runs experiments for section 4.1, testing prior predictive diversity

is_save = False
is_norm = True # whether to normalise cmap of filters per filter
filter_size = 5
sigma_g=0.2 # 0.3 for grayscale
filter_layer=0
# is_colour=False
# dataset_str = 'mnist' # fash_mnist mnist
# n_conv_layers=1
n_runs=500 # 100 500


# for dataset_str, n_conv_layers, sigma_g in [('mnist',1,0.),('mnist',1,0.2),('fash_mnist',1,0.),('fash_mnist',1,0.2),('mnist',3,0.),('fash_mnist',3,0.)]:
# for dataset_str, n_conv_layers, sigma_g in [('mnist',1,0.),('mnist',2,0.),('mnist',3,0.),('fash_mnist',1,0.),('fash_mnist',2,0.),('fash_mnist',3,0.)]:
# for dataset_str, n_conv_layers, sigma_g in [('cifar',1,0.),('cifar',2,0.),('cifar',3,0.)]:
for dataset_str, n_conv_layers, sigma_g in [('mnist',1,0.)]:
	x_train, y_train, x_test, y_test, input_shape = get_data(dataset_str)
	if dataset_str=='cifar':
		input_shape=(32,32,3)
	else:
		input_shape=(28,28,1)

	ents_all=[]
	for method_loop in ['rand','gabor_noise','fc']: # rand gabor blank gabor_noise
	# for method_loop in ['gabor']:
		print('\n',method_loop)
		ent_loop = []
		for i in range(n_runs):
			print(i,end='\r')

			# we used to forget to create a new model each loop actually this is v important
			K.clear_session()
			if 'fc' in method_loop:
				model = create_fc_model(n_hidden=1024,n_layers=n_conv_layers,input_shape=input_shape)
			else:
				model = create_model(filter_size=filter_size,n_conv_layers=n_conv_layers,n_first_filters=16,input_shape=input_shape)
				# model = create_lenet(filter_size=filter_size,n_conv_layers=n_conv_layers,n_first_filters=16,input_shape=input_shape)
				curr_filters = get_1st_filters(model)
				new_filters = create_1st_filters(curr_filters, method_in=method_loop,sigma_g=sigma_g) 
				model = set_1st_filters(model, new_filters)

			# print(model.summary())
			# filter_visualise(model,title_in=method_loop+'sig'+str(sigma_g)+'col'+str(is_colour),norm=is_norm,filter_layer=filter_layer,is_save=is_save,n_filters=32)

			rand_ix = np.random.randint(0,10)
			ent = predict_classes(model,x_test[rand_ix:rand_ix+500],'new random model',show_plot=False)
			ent_loop.append(ent)

			title_in=dataset_str+'_noise'+str(round(sigma_g,2)).replace('.','')+'_convs'+str(n_conv_layers)
			# ent = predict_classes(model,x_test,title_in=title_in,show_plot=True,method_in=method_loop,is_save=is_save)

		ent_loop=np.array(ent_loop)
		print(dataset_str,' n_conv_layers',n_conv_layers,' sigma_g',sigma_g)
		print('\nent for ',method_loop, ' mean =',round(ent_loop.mean(),3))
		print('ent for ',method_loop, ' std err =',round(ent_loop.std()/np.sqrt(n_runs),3))
		ents_all.append([method_loop,ent_loop])

	if False:
		fig, ax = plt.subplots(figsize=(4,4))
		# ax.set_title('Entropies on '+dataset_str+', f_size='+str(filter_size)+', sigma_g='+str(round(sigma_g,2))+', convs='+str(n_conv_layers))
		bins_list = np.arange(0.,2.1,0.1) 
		for method_loop,ent_loop in ents_all:
			if method_loop=='gabor_noise':
				label = 'Gabor $\sigma_g='+str(round(sigma_g,2))+'$'
			elif method_loop=='rand':
				label = 'i.i.d.'
			ax.hist(ent_loop,alpha=0.5,label=label ,color=col_dic[method_loop],bins=bins_list,density=True)
			# ax.hist(ent_loop,alpha=0.5,color=col_dic[method_loop],bins=bins_list)
			ax.axvline(ent_loop.mean(),ls='--',color=col_dic[method_loop],lw=2)

		fig.legend(loc=(0.15,0.75))
		ax.set_xlim([-0.05,2.05])
		ax.set_xlabel('Entropy')
		ax.set_ylabel('Frequency')
		# ax.set_title('training runs '+dataset_str+' fsize='+str(filter_size)+' convs='+str(n_conv_layers))
		fig.show()
		if is_save:
			print('saving...')
			fig.savefig('00_outputs/entropy_'+dataset_str+'_noise'+str(round(sigma_g,2)).replace('.','')+'_convs'+str(n_conv_layers)+time.strftime("%Y%m%d_%H%M%S")+'.pdf', format='pdf', dpi=500, bbox_inches='tight')



