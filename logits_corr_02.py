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

# this script runs experiments for section 4.2, testing prior activation correlations

# is_save = False
is_norm = True # whether to normalise cmap of filters per filter
filter_size = 5
sigma_g=0. 
filter_layer=0
# is_colour=False
dataset_str = 'cifar' # fash_mnist mnist cifar
n_conv_layers=1
method_loop='rand' # rand gabor_noise
# n_runs=5 # 100 500
n_egs=50
n_prior_draws=50
is_save=False

x_train, y_train, x_test, y_test, input_shape = get_data(dataset_str) # mnist fash_mnist

if dataset_str == 'mnist':
	data_labels=[0,1,2,3,4,5,6,7,8,9]
elif dataset_str == 'fash_mnist':
	data_labels=['tshirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','boot']
elif dataset_str == 'cifar':
	data_labels=['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']


if dataset_str=='cifar':
	input_shape=(32,32,3)
else:
	input_shape=(28,28,1)

# get some examples of each class
egs_dict = get_class_egs(x_test,y_test,n_egs)


# for method_loop, n_conv_layers in [('rand',1),('rand',2),('rand',3),('gabor_noise',1),('gabor_noise',2),('gabor_noise',3)]:
for method_loop, n_conv_layers in [('fc',1),('fc',2),('fc',3)]:
	# for these examples, run them through a NN drawn from prior
	egs_logits_dict={}
	for i in range(n_prior_draws):
		print('prior draw', i, end='\r')
		K.clear_session()
		
		# get new model
		if 'fc' in method_loop:
			model = create_fc_model(n_hidden=1024,n_layers=n_conv_layers,input_shape=input_shape)
		else:
			model = create_model(filter_size=filter_size,n_conv_layers=n_conv_layers,n_first_filters=16,input_shape=input_shape)
			# model = create_lenet(filter_size=filter_size,n_conv_layers=n_conv_layers,n_first_filters=16,input_shape=input_shape)
			
			# set first layer filters according to method
			curr_filters = get_1st_filters(model)
			new_filters = create_1st_filters(curr_filters, method_in=method_loop,sigma_g=sigma_g) 
			model = set_1st_filters(model, new_filters)

		for j in range(num_classes):
			if j not in egs_logits_dict.keys():
				egs_logits_dict[j]=[]
			# only need one output of logits
			y_logits = predict_logits(model,egs_dict[j])[:,0]
			egs_logits_dict[j].append(y_logits)

		# so this has logits of one output for all n_egs in all classes

	for i in range(num_classes):
		# egs_logits_dict is dict, with each item array of size [n_egs,n_prior_draws]
		egs_logits_dict[i] = np.array(egs_logits_dict[i]).T

	# plot pairs of examples
	for class1_id in [1,7]: # which class to use [1,7] fashion [1,8] for mnist
		for class2_id in [1,7]:
			for i in range(3):
				eg1_id = np.random.randint(n_egs) # which eg of this class to use
				eg2_id = np.random.randint(n_egs)
				while eg2_id == eg1_id and class1_id==class2_id:
					eg2_id = np.random.randint(n_egs)

				title_in=dataset_str+'_'+method_loop+'_sigg'+str(round(sigma_g,2)).replace('.','')+'_convs'+str(n_conv_layers) + 'class12_'+str(class1_id)+str(class2_id)+str(i)
				# plot_scatter_corr_egs(egs_logits_dict, egs_dict,class1_id,class2_id,eg1_id,eg2_id,data_labels,method_loop,title_in,is_save=is_save)


	# now more qualitatively find average correlation between images
	corr_matrix = np.zeros((num_classes,num_classes))
	for class1_id in range(num_classes):
		for class2_id in range(num_classes):
			print(class2_id,end='\r')
			corrs=[]
			for eg1_id in range(n_egs):
				for eg2_id in range(n_egs):
					if not (class1_id==class2_id and eg1_id==eg2_id):
						x_logits = egs_logits_dict[class1_id][eg1_id]
						y_logits = egs_logits_dict[class2_id][eg2_id]
						corrs.append(np.corrcoef(x_logits,y_logits)[0,1])
			corrs = np.array(corrs)		
			corr_matrix[class1_id,class2_id] = corrs.mean()
	
	# a bit hard to count up exactly how many input pairs contribute to the score
	# but if n_egs=50, I think it's about 50*49*10classes/2repeats=12,250
	# for the same class, and probably more for the different classes


	avg_same_class = np.diag(corr_matrix).mean()
	corrs_upper = np.triu(corr_matrix,1)
	avg_diff_class = corrs_upper.sum()/np.count_nonzero(corrs_upper)
	print(title_in)
	print('avg_same_class',round(avg_same_class,2))
	print('avg_diff_class',round(avg_diff_class,2))

	fig, ax = plt.subplots(figsize=(4.5,4.5))
	im = ax.imshow(corr_matrix,cmap='plasma',vmin=0.4,vmax=1.0)
	# text = ax.text(0.05, 0.05, 'Mean corr. same class='+'{:.2f}'.format(round(avg_same_class,2)), color='k',transform=ax.transAxes)
	# text = ax.text(0.05, 0.05, 'Mean corr. diff class='+'{:.2f}'.format(round(avg_diff_class,2)), color='k',transform=ax.transAxes)
	ax.set_xticks(np.arange(len(data_labels)))
	ax.set_yticks(np.arange(len(data_labels)))
	ax.set_xticklabels(data_labels)
	ax.set_yticklabels(data_labels)
	plt.setp(ax.get_xticklabels(), rotation=45, ha='right',rotation_mode='anchor')
	for i in range(len(data_labels)):
		for j in range(len(data_labels)):
			text = ax.text(j, i, '{:.2f}'.format(round(corr_matrix[i, j],2)),ha='center', va='center', color='k')
	# ax.set_title('output correlations on ' + dataset_str +', '+method_loop)

	# add in an example image of each class
	def offset_image(coord, img, ax, which_axis):
		im = OffsetImage(img, zoom=0.65,cmap='gray')
		im.image.axes = ax
		# ab = AnnotationBbox(im, (coord, 9),  xybox=(0., -25), frameon=False,xycoords='data',  boxcoords="offset points", pad=0)
		if which_axis=='x':
			xy=(coord*0.1 +0.05,-0.04)
		elif which_axis=='y':
			xy=(-0.05,0.95-coord*0.1)
		ab = AnnotationBbox(im, xy, xycoords='axes fraction',bboxprops={"edgecolor" : "none"})
		ax.add_artist(ab)
		return

	ax.tick_params(axis='x', which='major', pad=26)
	ax.tick_params(axis='y', which='major', pad=26)
	for i in range(0,num_classes):
		offset_image(i, egs_dict[i][0,:,:,0], ax,'x')
		offset_image(i, egs_dict[i][0,:,:,0], ax,'y')

	fig.tight_layout()
	fig.show()

	if is_save:
		print('saving...')
		title_in = dataset_str+'_'+method_loop+'_sigg'+str(round(sigma_g,2)).replace('.','')+'_convs'+str(n_conv_layers)+'_'
		# fig.savefig('00_outputs/01_logits_corr/logitscorr_'+title_in+time.strftime("%Y%m%d_%H%M%S")+'.pdf', format='pdf', dpi=500, bbox_inches='tight')
		fig.savefig('00_outputs/01_logits_corr/matrixlogitscorr_'+title_in+time.strftime("%Y%m%d_%H%M%S")+'.pdf', format='pdf', dpi=500, bbox_inches='tight')


