import importlib
import utils
importlib.reload(utils)
from utils import *
from sklearn.manifold import TSNE

# avoid the dreaded type 3 fonts...
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 10})
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = True

K.clear_session()

# this script runs experiments for section 4.3, testing cappa

tsne_col_dict={}
tsne_col_dict[0]='lawngreen'
tsne_col_dict[1]='dodgerblue'
tsne_col_dict[2]='tomato'
tsne_col_dict[3]='gray'
tsne_col_dict[4]='cyan'
tsne_col_dict[5]='lightpink'
tsne_col_dict[6]='orange'
tsne_col_dict[7]='k'
tsne_col_dict[8]='sienna'
tsne_col_dict[9]='violet'


is_save = False
is_plot = False
# is_norm = True # whether to normalise cmap of filters per filter
filter_size = 5
n_egs = 1000 # 200
n_tsne_class=2 # how many classes to plot on this
# n_conv_layers=1
# method_loop='gabor_noise' # rand gabor gabor_noise
sigma_g=0.
dataset_str = 'fash_mnist' # mnist fash_mnist cifar
x_train, y_train, x_test, y_test, input_shape = get_data(dataset_str)

if dataset_str=='cifar':
    input_shape=(32,32,3)
else:
    input_shape=(28,28,1)
data_labels = data_labels_dict[dataset_str]

start_idx=0
egs_dict = get_class_egs(x_train[start_idx:],y_train[start_idx:],n_egs)

# use an offset to select a pair of slightly more different classes
# (cifar default is car and plane)
offset=0
if dataset_str=='cifar':
    offset=1

x_small = np.zeros((n_tsne_class*n_egs,)+egs_dict[0].shape[1:])
y_small = np.zeros((n_tsne_class*n_egs,))
# for i in range(n_tsne_class):
for i in range(offset,n_tsne_class+offset):
    x_small[(i-offset)*n_egs:((i-offset)+1)*n_egs] = egs_dict[i]
    y_small[(i-offset)*n_egs:((i-offset)+1)*n_egs] = i-offset # this will still be 0 and 1

if is_plot:
    # fit TSNE embedding model
    print('\n-- starting tsne...\n')
    tsne_model = TSNE(n_components=2, perplexity=50,learning_rate=100,random_state=0)
    tsne_embed = tsne_model.fit_transform(x_small.reshape(x_small.shape[0],-1))
    print('\n-- finished tsne...\n')

if False:
    fig, ax = plt.subplots(figsize=(4,4))
    for i in range(n_tsne_class):
        ax.scatter(tsne_embed[i*n_egs:(i+1)*n_egs,0],tsne_embed[i*n_egs:(i+1)*n_egs,1],edgecolors='k',linewidths=0.5,s=50)
    fig.show()


for n_conv_layers in [1,2,3]:
    result={} # store cappa per method
    methods=['iid fc','iid conv','gabor conv']
    # methods=['iid conv']
    # methods=['iid fc']
    # for method_loop in ['rand','gabor_noise']:
    # for method_loop, n_conv_layers in [('iid fc',1),('iid conv',1),('gabor conv',1),('iid fc',2),('iid conv',2),('gabor conv',2)]:
    # for method_loop, n_conv_layers in [('iid fc',1)]:
    for method_loop in methods:
        print('\n--working on ',method_loop)
        n_runs = 500
        cols = 4
        rows = n_runs/cols
        if is_plot: fig = plt.figure(figsize=(min(cols*1.5,6),min(rows*1.5,6)))
        for run in range(n_runs):
            print('run',run,end='\r')
            # create model w draw from prior, and make predictions
            K.clear_session()
            if 'conv' in method_loop:
                # print('conv model')
                model = create_model(filter_size=filter_size,n_conv_layers=n_conv_layers,n_first_filters=16,input_shape=input_shape)
                curr_filters = get_1st_filters(model)
                if 'gabor' in method_loop:
                    method_filter='gabor_noise'
                else:
                    method_filter='rand'
                new_filters = create_1st_filters(curr_filters, method_in=method_filter,sigma_g=sigma_g) 
                model = set_1st_filters(model, new_filters)

            elif 'fc' in method_loop:
                # print('fc model')
                model = create_fc_model(n_hidden=1024,n_layers=n_conv_layers,input_shape=input_shape)
            else:
                raise(Exception)

            if run==0 and False:
                print(model.summary())

            # train_model(model,l_rate=l_rate,epochs_in=1,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,val_size=x_test.shape[0])
            y_preds = model.predict(x_small, verbose=0)

            # process the outputs. only want it to have as many outputs as classes
            y_preds = y_preds[:,:n_tsne_class]
            y_preds_class = np.argmax(y_preds,axis=-1)
            y_preds_cols = []
            for i in range(y_preds_class.shape[0]):
                y_preds_cols.append(tsne_col_dict[y_preds_class[i]])

            # compute class agnostic prior predictive accuracy (CAPPA)
            cappa = 0.
            # try all permutations of class assignations
            for i in range(n_tsne_class):
                if i==0:
                    y_small_test=y_small.copy()
                elif i==1:
                    y_small_test=((y_small-1)*-1).copy()
                acc = (y_preds_class==y_small_test).mean()
                if acc>cappa: cappa=acc
            # print('cappa',cappa)

            if method_loop not in result.keys(): result[method_loop]=[]
            result[method_loop].append(cappa)


            if False:
                # plots actual assignment
                fig, ax = plt.subplots(figsize=(4,4))
                # title_in='draw from '+method_loop.replace('_','')+',sample'+str(run)
                # ax.set_title(title_in)
                for i in range(n_tsne_class):
                    ax.scatter(tsne_embed[i*n_egs:(i+1)*n_egs,0],tsne_embed[i*n_egs:(i+1)*n_egs,1],edgecolors='k',linewidths=0.5,s=50)
                # ax.scatter(tsne_embed[:,0],tsne_embed[:,1],c=y_preds_cols,edgecolors='k',linewidths=0.5,s=50)
                # # ax.scatter(tsne_embed[:,0],tsne_embed[:,1],c=y_preds_class)
                # ax.set_xlabel('Latent dim 1')
                # ax.set_ylabel('Latent dim 2')
                # ax.set_yticklabels([])
                # ax.set_yticks([])
                # ax.set_xticklabels([])
                # ax.set_xticks([])
                # fig.tight_layout()
                fig.show()

            if is_plot: 
                ax = fig.add_subplot(rows, cols, run+1)
                if run==1:
                    title_in=str(n_runs)+' prior draw from '+method_loop.replace('_','')+'NN'
                    ax.set_title(title_in)
                ax.set_yticklabels([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.scatter(tsne_embed[:,0],tsne_embed[:,1],c=y_preds_cols,edgecolors='k',linewidths=0.2,s=15)
                text = ax.text(0.5, 0.1, '{:.2f}'.format(round(cappa,2)),ha='center', va='center', color='k',transform=ax.transAxes)

        # fig.tight_layout()
        if is_plot: fig.show()
        
    result['baseline']=[]
    for i in range(1000):
        y_preds_class = np.random.binomial(n=1,p=0.5,size=y_small.shape)
        cappa = 0.
        for i in range(n_tsne_class):
            if i==0:
                y_small_test=y_small.copy()
            elif i==1:
                y_small_test=((y_small-1)*-1).copy()
            acc = (y_preds_class==y_small_test).mean()
            if acc>cappa: cappa=acc
        result['baseline'].append(cappa)
    # result['baseline']=np.array(result['baseline'])

    print('\n\nresults on ',dataset_str)
    print('n_conv_layers',n_conv_layers)
    for key in result.keys():
        result[key]=np.array(result[key])
        print('cappa',key, round(result[key].mean(),3),' pm ',round(result[key].std()/np.sqrt(n_runs),3))



    if False:
        # visualise
        fig, ax = plt.subplots(figsize=(4,4))
        for i in range(n_tsne_class):
            ax.scatter(tsne_embed[i*n_egs:(i+1)*n_egs,0],tsne_embed[i*n_egs:(i+1)*n_egs,1])
        fig.tight_layout()
        fig.show()
        
    if is_save:
        print('saving...')
        # title_in = dataset_str+'_'+method_loop+'_sigg'+str(round(sigma_g,2)).replace('.','')+'_convs'+str(n_conv_layers)+'_'
        # fig.savefig('00_outputs/01_logits_corr/logitscorr_'+title_in+time.strftime("%Y%m%d_%H%M%S")+'.pdf', format='pdf', dpi=500, bbox_inches='tight')
        # fig.savefig('00_outputs/01_logits_corr/matrixlogitscorr_'+title_in+time.strftime("%Y%m%d_%H%M%S")+'.pdf', format='pdf', dpi=500, bbox_inches='tight')








