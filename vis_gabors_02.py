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

# this script displays egs of the prob gabor filter, used for figure 2

is_save = False
is_norm = True # whether to normalise cmap of filters per filter
method_loop = 'gabor_noise' # blank, rand, gabor, gabor_noise
filter_size = 2
sigma_g=0.
filter_layer=0
is_colour=False

# for filter_size, is_colour, sigma_g in [(5,False,0.),(5,False,0.3)]:
for filter_size, is_colour, sigma_g in [(17,True,0.)]: # twitter
# for filter_size, is_colour, sigma_g in [(3,True,0.),(3,True,0.1),(7,True,0.),(7,True,0.1)]: # paper
    if is_colour:
        input_shape=(28,28,3)
    else:
        input_shape=(28,28,1)

    model = create_lenet(filter_size=filter_size,n_conv_layers=1,n_first_filters=32,input_shape=input_shape)
    # print(model.summary())

    curr_filters = get_1st_filters(model)
    new_filters = create_1st_filters(curr_filters, method_in=method_loop,sigma_g=sigma_g) 
    model = set_1st_filters(model, new_filters)

    filter_visualise(model,title_in=method_loop+'sig'+str(sigma_g)+'col'+str(is_colour),norm=is_norm,filter_layer=filter_layer,is_save=is_save,n_filters=32) # paper
    # filter_visualise(model,title_in=method_loop+'sig'+str(sigma_g)+'col'+str(is_colour),norm=is_norm,filter_layer=filter_layer,is_save=is_save,rows_in=16,n_filters=512) # twitter


# grayscale used these
# sigma =  np.random.uniform(2,10) # think to do with scale of whole thing? was 0.5,10
# theta =  np.random.uniform(0,3.14) # orientation of waves
# Lambda = np.random.uniform(3,filter_width*2) # 1/freq of waves was 1 to *1
# psi =    np.random.uniform(-3.14,3.14) # phase offset..?
# gamma =  np.random.uniform(0.,3) # ellipticity was 0.2 1.5

# colour used these
# sigma =  np.random.uniform(2,4) # think to do with scale of whole thing? was 0.5,10
# theta =  np.random.uniform(0,3.14) # orientation of waves
# Lambda = np.random.uniform(2,10) # 1/freq of waves 
# psi =    np.random.uniform(-3.14,3.14) # phase offset..?
# gamma =  np.random.uniform(0.,3) # ellipticity was 0.2 1.5

