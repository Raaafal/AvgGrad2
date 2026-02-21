import copy
import math

import matplotlib.pyplot as plt
import utils
import json
import numpy as np
import statistics
from matplotlib.ticker import MaxNLocator
from scipy import stats
from utils import normality_test


# plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


plot_style = {
    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage{amsmath} "
        r"\usepackage{bm} "
        r"\boldmath "  # Bolds all math
        r"\renewcommand{\seriesdefault}{\bfdefault}"  # Bolds all regular text
    ),
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],

    # Sizes
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
}

plt.rcParams.update(plot_style)


directory='plot_data2/'
save_directory='saved/'
name='Our Algorithm'
save_format='pdf'#'png'
save_plots=True
delta_loss_default_range_linear_scale=(0,1000)
accuracy_range={'x':None,'y':None}
train_loss_range={'x':None,'y':None}
test_loss_range={'x':None,'y':None}

epoch_ratio_bar_plot=False

plot_test_loss=True
plot_test_acc=True
plot_relative_loss_improvements=True

log_scale=False
args={'figsize':(5,5),'dpi':100}

saved_dpi=1000

plot_means=True #only for test loss
plot_means_err=True
plot_medians=False
plot_medians_err=False

test_normality=False

train_avg_relative_loss_improvement_how_many_outliers_to_exclude=0

means_linestyle='--'
medians_linestyle='-'
if not plot_medians:
    means_linestyle='-'

if plot_means and not plot_medians:
    means_linestyle='-'

plot_means_args={'linewidth':1.5}#{'linewidth':.5}

legend_linewidth=1.33333#1.33333

colors_colorblind_safe6 = ["#0072B2", "#D55E00", "#56B4E9", "#CC79A7", "#009E73", "#E69F00"]#["#0072B2", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]
colors_colorblind_safe6 = ["#0072B2", "#D55E00", "#009E73", "#E69F00","#56B4E9", "#CC79A7"]#["#0072B2", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]
colors_colorblind_safe7 = [
    "#0072B2", # Blue
    "#D55E00", # Vermilion
    "#56B4E9", # Sky Blue
    "#CC79A7", # Reddish Purple
    "#009E73", # Bluish Green
    "#E69F00", # Orange
    "#F0E442"  # Yellow
]

sample_efficiency_comparison_metric='train_loss'#'train_loss_aggregated'
method1_sample_efficiency_comparison='None'
method2_sample_efficiency_comparison='None'

# not_improved_file_name='results_model5_not_improved_training_seed23_200trainings_30e_not_mnist'
# improved_file_name='results_model5_improved_training_seed23_100trainings_30e_not_mnist'
# improved_more_iter_file_name='results_model5_improved_training_iter5_seed23_50trainings_30e_not_mnist'

# not_improved_file_name='results_model6_not_improved_training_seed23_200trainings_40e'
# improved_file_name='results_model6_improved_training_seed23_100trainings_40e'
# improved_more_iter_file_name='results_model6_improved_training_iter5_seed23_50trainings_40e'
# delta_loss_default_range_linear_scale=(4,1000)

# not_improved_file_name='results_model6_not_improved_training_seed23_200trainings_40e_not_mnist'
# improved_file_name='results_model6_improved_training_seed23_100trainings_40e_not_mnist'
# improved_more_iter_file_name='results_model6_improved_training_iter5_seed23_50trainings_40e_not_mnist'
# delta_loss_default_range_linear_scale=(12,1000)

# not_improved_file_name='results_model5_not_improved_training_seed0_200trainings_30e'
# improved_file_name='results_model5_improved_training_seed24_100trainings_30e'
# improved_more_iter_file_name='results_model5_improved_training_iter5_seed23_50trainings_30e'

#0.1 lr
# not_improved_file_name='results_model5_not_improved_training_seed23_200trainings_30e_0.1lr'
# improved_file_name='results_model5_improved_training_seed23_100trainings_30e_0.1lr'
# improved_more_iter_file_name='results_model5_improved_training_iter5_seed23_100trainings_30e_0.1lr'

# not_improved_file_name='results_model6_not_improved_training_seed23_200trainings_40e_0.1lr'
# improved_file_name='results_model6_improved_training_iter2_seed23_100trainings_40e_0.1lr'
# improved_more_iter_file_name='results_model6_improved_training_iter5_seed23_50trainings_40e_0.1lr'
# delta_loss_default_range_linear_scale=(20,1000)

# not_improved_file_name='results_model5_not_improved_training_seed23_200trainings_30e_not_mnist_0.1lr'
# improved_file_name='results_model5_improved_training_iter2_seed23_100trainings_30e_not_mnist_0.1lr'
# improved_more_iter_file_name='results_model5_improved_training_iter5_seed23_100trainings_30e_not_mnist_0.1lr'

# not_improved_file_name='results_model6_not_improved_training_seed23_200trainings_40e_not_mnist_0.1lr'
# improved_file_name='results_model6_improved_training_iter2_seed23_100trainings_40e_not_mnist_0.1lr'
# improved_more_iter_file_name='results_model6_improved_training_iter5_seed23_50trainings_40e_not_mnist_0.1lr'
######################
# improved_more_iter_file_name=None

### not_improved_file_name='results_model5_not_improved_training_seed0_200trainings_30e'
### improved_file_name='results_model5_improved_training_iter5_seed23_50trainings_30e'

# # not_improved_file_name='results_model5_not_improved_training_seed23_100trainings_30e_not_mnist'
# # improved_file_name='results_model5_improved_training_iter2_seed23_100trainings_30e_not_mnist'
# # improved_more_iter_file_name='results_model5_improved_training_iter5_seed23_50trainings_30e_not_mnist'
# #
# # not_improved_file_name='results_model6_not_improved_training_seed23_100trainings_40e'
# # improved_file_name='results_model6_improved_training_iter2_seed23_100trainings_40e'
# # improved_more_iter_file_name='results_model6_improved_training_iter5_seed23_50trainings_40e'
# # delta_loss_default_range_linear_scale=(4,1000)
# #
# # not_improved_file_name='results_model6_not_improved_training_seed23_100trainings_40e_not_mnist'
# # improved_file_name='results_model6_improved_training_iter2_seed23_100trainings_40e_not_mnist'
# # improved_more_iter_file_name='results_model6_improved_training_iter5_seed23_50trainings_40e_not_mnist'
# # delta_loss_default_range_linear_scale=(12,1000)
# #
# # not_improved_file_name='results_model5_not_improved_training_seed23_100trainings_30e'
# # improved_file_name='results_model5_improved_training_iter2_seed23_100trainings_30e'
# # improved_more_iter_file_name='results_model5_improved_training_iter5_seed23_50trainings_30e'
# #
# # 0.1 lr
# # not_improved_file_name='results_model5_not_improved_training_seed23_100trainings_30e_0.1lr'
# # improved_file_name='results_model5_improved_training_iter2_seed23_100trainings_30e_0.1lr'
# # improved_more_iter_file_name='results_model5_improved_training_iter5_seed23_50trainings_30e_0.1lr'
# #
# # not_improved_file_name='results_model6_not_improved_training_seed23_100trainings_40e_0.1lr'
# # improved_file_name='results_model6_improved_training_iter2_seed23_100trainings_40e_0.1lr'
# # improved_more_iter_file_name='results_model6_improved_training_iter5_seed23_50trainings_40e_0.1lr'
# # delta_loss_default_range_linear_scale=(20,1000)
# #
# # not_improved_file_name='results_model5_not_improved_training_seed23_100trainings_30e_not_mnist_0.1lr'
# # improved_file_name='results_model5_improved_training_iter2_seed23_100trainings_30e_not_mnist_0.1lr'
# # improved_more_iter_file_name='results_model5_improved_training_iter5_seed23_50trainings_30e_not_mnist_0.1lr'
# #
# # not_improved_file_name='results_model6_not_improved_training_seed23_100trainings_40e_not_mnist_0.1lr'
# # improved_file_name='results_model6_improved_training_iter2_seed23_100trainings_40e_not_mnist_0.1lr'
# # improved_more_iter_file_name='results_model6_improved_training_iter5_seed23_50trainings_40e_not_mnist_0.1lr'

######################################################################################
# not_improved_file_name='results_model6_not_improved_training_seed5_200trainings_15e'
# improved_file_name='results_model6_improved_training_iter2_seed5_100trainings_15e'
# improved_more_iter_file_name='results_model6_improved_training_iter5_seed5_100trainings_15e'
# delta_loss_default_range_linear_scale=(4,1000)

# not_improved_file_name='results_model6_not_improved_training_seed5_200trainings_15e_not_mnist'
# improved_file_name='results_model6_improved_training_iter2_seed5_100trainings_15e_not_mnist'
# improved_more_iter_file_name='results_model6_improved_training_iter5_seed5_100trainings_15e_not_mnist'

# accuracy_range={'x':[100,510],'y':[90,98]}
# not_improved_file_name='results_model9_not_improved_training_seed4_15trainings_500e'
# improved_file_name='results_model9_improved_training_iter2_seed4_15trainings_300e'
# improved_more_iter_file_name='results_model9_improved_training_iter5_seed4_7trainings_300e'

# accuracy_range={'x':[100,510],'y':[80.5,88.5]}
# not_improved_file_name='results_model9_not_improved_training_seed4_15trainings_500e_not_mnist'
# improved_file_name='results_model9_improved_training_iter2_seed4_15trainings_300e_not_mnist'
# improved_more_iter_file_name='results_model9_improved_training_iter5_seed4_7trainings_300e_not_mnist'

###############plots for optimal LR for the gradient-based RMSProp:
# not_improved_file_name='results_model9_not_improved_training_seed4_15trainings_500e'
# improved_file_name='results_model9_improved_training_iter2_seed6_3trainings_500e'
# improved_more_iter_file_name='results_model9_improved_training_iter5_seed6_2trainings_500e'

# not_improved_file_name='results_model9_not_improved_training_seed4_15trainings_500e_not_mnist'
# improved_file_name='results_model9_improved_training_iter2_seed6_3trainings_500e_not_mnist'
# improved_more_iter_file_name='results_model9_improved_training_iter5_seed6_2trainings_500e_not_mnist'

#IMDB
# # not_improved_file_name='results_model12_not_improved_training_seed21_30trainings_200e_imdb_larger_lr'#'results_model12_not_improved_training_seed21_15trainings_200e_imdb_larger_lr'
# not_improved_file_name='results_model12_not_improved_training_seed12_30trainings_200e_imdb'
# improved_file_name='results_model12_improved_training_iter2_seed11_15trainings_150e_imdb'
#
# improved_more_iter_file_name='results_model12_improved_training_iter2_seed11_15trainings_150e_imdb'

#IMDB 2 good comparison
###not_improved_file_name='results_model12_not_improved_training_seed21_30trainings_200e_imdb_larger_lr'#'results_model12_not_improved_training_seed21_15trainings_200e_imdb_larger_lr'
# not_improved_file_name='results_model12_not_improved_training_seed21_50trainings_200e_imdb_larger_lr'
# improved_file_name='results_model12_improved_training_iter2_seed14_30trainings_200e_imdb_larger_lr'
# improved_more_iter_file_name='results_model12_improved_training_iter4_seed15_15trainings_200e_imdb_larger_lr'

# #IMDB 3 best comparison
# not_improved_file_name='results_model12_not_improved_training_seed12_30trainings_200e_imdb'
# improved_file_name='results_model12_improved_training_iter2_seed14_30trainings_200e_imdb_larger_lr'
#
# improved_more_iter_file_name='results_model12_improved_training_iter4_seed15_15trainings_200e_imdb_larger_lr'


# not_improved=utils.read_file(directory+not_improved_file_name+'.txt')
# improved=utils.read_file(directory+improved_file_name+'.txt')
# improved_more_iter=utils.read_file(directory+improved_more_iter_file_name+'.txt') if improved_more_iter_file_name is not None else None

###############################################################################################################3
data_to_plot=[]

# # #model 6 FASHION MNIST
# # # data_to_plot.append(utils.PlotStatistics('Soap w/o Momentum',"results_model6_not_improved_training20_seed3_50trainings_15e_fashion_mnist_SOAP_betas(0.0,0.95)_batch_size128",'darkcyan',plot_color_median='darkblue'))
# # # data_to_plot.append(utils.PlotStatistics('Soap',"results_model6_not_improved_training20_seed3_50trainings_15e_fashion_mnist_SOAP_betas(0.95,0.95)_batch_size128",'magenta',plot_color_median='darkmagenta'))
# # # data_to_plot.append(utils.PlotStatistics('RMSProp AG-1 Linear',"results_model6_improved_training21_seed3_25trainings_15e_fashion_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size128",'limegreen',plot_color_median='darkgreen'))
# # # data_to_plot.append(utils.PlotStatistics('Soap AG-2 Linear',"results_model6_improved_training22_seed3_50trainings_15e_fashion_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'red',plot_color_median='darkred'))
# # # data_to_plot.append(utils.PlotStatistics('Soap AG-2',"results_model6_improved_training22_seed3_50trainings_15e_fashion_mnist__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'black',plot_color_median='black'))
# #
# # data_to_plot.append(utils.PlotStatistics('Soap w/o Momentum',"results_model6_not_improved_training20_seed3_50trainings_15e_fashion_mnist_SOAP_betas(0.0,0.95)_batch_size128",'darkcyan',plot_color_median='darkblue'))
# data_to_plot.append(utils.PlotStatistics('Soap',"results_model6_not_improved_training20_seed3_50trainings_15e_fashion_mnist_SOAP_betas(0.95,0.95)_batch_size128",'magenta',plot_color_median='darkmagenta'))
#
# data_to_plot.append(utils.PlotStatistics('RMSProp',"results_model6_not_improved_training_seed5_200trainings_15e_not_mnist",'brown',plot_color_median='brown'))
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-1',"results_model6_improved_training_iter2_seed5_100trainings_15e_not_mnist",'purple',plot_color_median='purple'))
#
# data_to_plot.append(utils.PlotStatistics('Adam',"adam_results_model6_not_improved_training_seed16_50trainings_15e_fashion_mnist",'aquamarine',plot_color_median='aquamarine'))
#
# # data_to_plot.append(utils.PlotStatistics('RMSProp AG-1 Linear',"results_model6_improved_training21_seed3_25trainings_15e_fashion_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size128",'limegreen',plot_color_median='darkgreen'))
# # #data_to_plot.append(utils.PlotStatistics('RMSProp AG-2',"",'orange',plot_color_median='orange'))
# # # data_to_plot.append(utils.PlotStatistics('m22 Soap AG-2 Linear',"results_model6_improved_training22_seed3_50trainings_15e_fashion_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'yellow',plot_color_median='yellow'))
# # # data_to_plot.append(utils.PlotStatistics('m22 Soap AG-2',"results_model6_improved_training22_seed3_50trainings_15e_fashion_mnist__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'pink',plot_color_median='pink'))
#
# # data_to_plot.append(utils.PlotStatistics('Soap AG-2 Linear',"results_model6_improved_training27_seed3_30trainings_15e_fashion_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'red',plot_color_median='darkred'))
# data_to_plot.append(utils.PlotStatistics('Soap AG-2',"results_model6_improved_training27_seed3_30trainings_15e_fashion_mnist__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'black',plot_color_median='black'))
#
# data_to_plot.append(utils.PlotStatistics('Soap AG-4',"results_model6_improved_training30_seed4_50trainings_15e_fashion_mnist__nonlinear_layers_avggrad_SOAP_betas(0.95,0.95)_batch_size128",'pink',plot_color_median='pink'))
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-4',"results_model6_improved_training30_seed4_50trainings_15e_fashion_mnist__nonlinear_layers_avggrad_RMSprop_batch_size128",'darkred',plot_color_median='darkred'))
#
#
#
# which_method_for_ratio_of_all_batches='AG'
# method1_sample_efficiency_comparison='RMSProp AG-1'#'Adam'
# #method1_sample_efficiency_comparison='Adam'
# method2_sample_efficiency_comparison='RMSProp'

# #model 6 MNIST
# # # data_to_plot.append(utils.PlotStatistics('Soap w/o Momentum',"results_model6_not_improved_training20_seed3_50trainings_15e_mnist_SOAP_betas(0.0,0.95)_batch_size128",'darkcyan',plot_color_median='darkblue'))
# # # data_to_plot.append(utils.PlotStatistics('Soap',"results_model6_not_improved_training20_seed3_50trainings_15e_mnist_SOAP_betas(0.95,0.95)_batch_size128",'magenta',plot_color_median='darkmagenta'))
# # # data_to_plot.append(utils.PlotStatistics('RMSProp AG-1 Linear',"results_model6_improved_training21_seed3_25trainings_15e_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size128",'limegreen',plot_color_median='darkgreen'))
# # # data_to_plot.append(utils.PlotStatistics('Soap AG-2 Linear',"results_model6_improved_training22_seed3_50trainings_15e_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'red',plot_color_median='darkred'))
# # # data_to_plot.append(utils.PlotStatistics('Soap AG-2',"results_model6_improved_training22_seed3_50trainings_15e_mnist__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'black',plot_color_median='black'))
# #
# # data_to_plot.append(utils.PlotStatistics('Soap w/o Momentum',"results_model6_not_improved_training20_seed3_50trainings_15e_mnist_SOAP_betas(0.0,0.95)_batch_size128",'darkcyan',plot_color_median='darkblue'))
# data_to_plot.append(utils.PlotStatistics('Soap',"results_model6_not_improved_training20_seed3_50trainings_15e_mnist_SOAP_betas(0.95,0.95)_batch_size128",'magenta',plot_color_median='darkmagenta'))
#
# data_to_plot.append(utils.PlotStatistics('RMSProp',"results_model6_not_improved_training_seed5_200trainings_15e",'brown',plot_color_median='brown'))
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-1',"results_model6_improved_training_iter2_seed5_100trainings_15e",'purple',plot_color_median='purple'))
#
# data_to_plot.append(utils.PlotStatistics('Adam',"adam_results_model6_not_improved_training_seed15_50trainings_15e_mnist",'aquamarine',plot_color_median='aquamarine'))
#
# # data_to_plot.append(utils.PlotStatistics('RMSProp AG-1 Linear',"results_model6_improved_training21_seed3_25trainings_15e_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size128",'limegreen',plot_color_median='darkgreen'))
# # # data_to_plot.append(utils.PlotStatistics('RMSProp AG-2',"",'orange',plot_color_median='orange'))
# # # data_to_plot.append(utils.PlotStatistics('m22 Soap AG-2 Linear',"results_model6_improved_training22_seed3_50trainings_15e_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'red',plot_color_median='darkred'))
# # # data_to_plot.append(utils.PlotStatistics('m22 Soap AG-2',"results_model6_improved_training22_seed3_50trainings_15e_mnist__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'black',plot_color_median='black'))
# # data_to_plot.append(utils.PlotStatistics('Soap AG-2 Linear',"results_model6_improved_training27_seed3_30trainings_15e_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'red',plot_color_median='darkred'))
# data_to_plot.append(utils.PlotStatistics('Soap AG-2',"results_model6_improved_training27_seed3_30trainings_15e_mnist__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'black',plot_color_median='black'))
#
# data_to_plot.append(utils.PlotStatistics('Soap AG-4',"results_model6_improved_training30_seed4_50trainings_15e_mnist__nonlinear_layers_avggrad_SOAP_betas(0.95,0.95)_batch_size128",'pink',plot_color_median='pink'))
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-4',"results_model6_improved_training30_seed4_50trainings_15e_mnist__nonlinear_layers_avggrad_RMSprop_batch_size128",'darkred',plot_color_median='darkred'))
#
#
# which_method_for_ratio_of_all_batches='AG'
# method1_sample_efficiency_comparison='RMSProp AG-1'#'Adam'
# #method1_sample_efficiency_comparison='Adam'
# method2_sample_efficiency_comparison='Adam'


# # #model 9 FASHION MNIST
# # data_to_plot.append(utils.PlotStatistics('Soap w/o Momentum',"results_model9_not_improved_training20_seed3_50trainings_50e_fashion_mnist_SOAP_betas(0.0,0.95)_batch_size128",'darkcyan',plot_color_median='darkblue'))
# # data_to_plot.append(utils.PlotStatistics('Soap',"results_model9_not_improved_training20_seed3_50trainings_50e_fashion_mnist_SOAP_betas(0.95,0.95)_batch_size128",'magenta',plot_color_median='darkmagenta'))
# # data_to_plot.append(utils.PlotStatistics('RMSProp AG-1 Linear',"results_model9_improved_training21_seed3_25trainings_50e_fashion_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size128",'limegreen',plot_color_median='darkgreen'))
# # data_to_plot.append(utils.PlotStatistics('Soap AG-2 Linear',"results_model9_improved_training22_seed3_50trainings_50e_fashion_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'red',plot_color_median='darkred'))
# # data_to_plot.append(utils.PlotStatistics('Soap AG-2',"results_model9_improved_training22_seed3_50trainings_50e_fashion_mnist__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'black',plot_color_median='black'))
# #
# # which_method_for_ratio_of_all_batches='AG'
#
# data_to_plot.append(utils.PlotStatistics('RMSProp',"results_model9_not_improved_training_seed12_50trainings_125e_fashion_mnist",colors_colorblind_safe6[1],plot_color_median=colors_colorblind_safe6[1]))
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-1',"results_model9_improved_training_iter2_seed10_8trainings_50e_fashion_mnist",colors_colorblind_safe6[2],plot_color_median=colors_colorblind_safe6[2]))
#
# # data_to_plot.append(utils.PlotStatistics('RMSProp AG-1 Linear',"results_model9_improved_training21_seed3_25trainings_50e_fashion_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size128",'limegreen',plot_color_median='darkgreen'))
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-2',"results_model9_improved_training27_seed3_30trainings_50e_fashion_mnist__nonlinear_layers_avggrad_RMSprop_batch_size128",colors_colorblind_safe7[6],plot_color_median=colors_colorblind_safe7[6]))
# # data_to_plot.append(utils.PlotStatistics('Soap AG-2 Linear',"results_model9_improved_training27_seed3_30trainings_50e_fashion_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'red',plot_color_median='darkred'))
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-3-1',"results_model9_improved_training30_seed4_50trainings_50e_fashion_mnist__nonlinear_layers_avggrad_RMSprop_batch_size128",colors_colorblind_safe6[5],plot_color_median=colors_colorblind_safe6[5]))
#
# data_to_plot.append(utils.PlotStatistics('Adam',"adam_results_model9_not_improved_training_seed17_15trainings_125e_fashion_mnist",colors_colorblind_safe6[3],plot_color_median=colors_colorblind_safe6[3]))
#
#
# data_to_plot.append(utils.PlotStatistics('Soap',"results_model9_not_improved_training20_seed3_50trainings_50e_fashion_mnist_SOAP_betas(0.95,0.95)_batch_size128",colors_colorblind_safe6[0],plot_color_median=colors_colorblind_safe6[0]))
# # data_to_plot.append(utils.PlotStatistics('Soap AG-3-1',"results_model9_improved_training30_seed4_50trainings_50e_fashion_mnist__nonlinear_layers_avggrad_SOAP_betas(0.95,0.95)_batch_size128",colors_colorblind_safe6[4],plot_color_median=colors_colorblind_safe6[4]))
#
# data_to_plot.append(utils.PlotStatistics('Soap ($\\beta_2=0$)',"results_model9_not_improved_training20_seed3_50trainings_50e_fashion_mnist_SOAP_betas(0.0,0.95)_batch_size128",colors_colorblind_safe6[4],plot_color_median=colors_colorblind_safe6[4]))
#
#
# data_to_plot.append(utils.PlotStatistics('Soap AG-2 ($\\beta_2=0$)',"results_model9_improved_training27_seed3_30trainings_50e_fashion_mnist__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'black',plot_color_median='black'))
#
# accuracy_range={'x':[0,127],'y':[80,89]}
# train_loss_range={'x':None,'y':[0.18,0.7]}
# test_loss_range={'x':None,'y':[0.33,0.6]}
#
# which_method_for_ratio_of_all_batches='AG'
# # method1_sample_efficiency_comparison='Soap AG-2'#'RMSProp AG-1'#'Adam'
# method1_sample_efficiency_comparison='Soap ($\\beta_2=0$)'
# # method2_sample_efficiency_comparison='Soap'
# method2_sample_efficiency_comparison='Soap AG-2 ($\\beta_2=0$)'


# # #model 9 MNIST
# #data_to_plot.append(utils.PlotStatistics('Soap w/o Momentum',"results_model9_not_improved_training20_seed3_50trainings_50e_mnist_SOAP_betas(0.0,0.95)_batch_size128",'darkcyan',plot_color_median='darkblue'))
#
# # # data_to_plot.append(utils.PlotStatistics('Soap',"results_model9_not_improved_training20_seed3_50trainings_50e_mnist_SOAP_betas(0.95,0.95)_batch_size128",'magenta',plot_color_median='darkmagenta'))
# # # data_to_plot.append(utils.PlotStatistics('RMSProp AG-1 Linear',"results_model9_improved_training21_seed3_25trainings_50e_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size128",'limegreen',plot_color_median='darkgreen'))
# # # data_to_plot.append(utils.PlotStatistics('Soap AG-2 Linear',"results_model9_improved_training22_seed3_50trainings_50e_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'red',plot_color_median='darkred'))
# # # data_to_plot.append(utils.PlotStatistics('Soap AG-2',"results_model9_improved_training22_seed3_50trainings_50e_mnist__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'black',plot_color_median='black'))
# # #
# # # which_method_for_ratio_of_all_batches='AG'
# #
# # data_to_plot.append(utils.PlotStatistics('Soap w/o Momentum',"results_model9_not_improved_training20_seed3_50trainings_50e_mnist_SOAP_betas(0.0,0.95)_batch_size128",'darkcyan',plot_color_median='darkblue'))
#
# data_to_plot.append(utils.PlotStatistics('RMSProp',"results_model9_not_improved_training_seed13_50trainings_125e_mnist",colors_colorblind_safe6[1],plot_color_median=colors_colorblind_safe6[1]))
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-1',"results_model9_improved_training_iter2_seed11_8trainings_50e_mnist",colors_colorblind_safe6[2],plot_color_median=colors_colorblind_safe6[2]))
#
# # data_to_plot.append(utils.PlotStatistics('RMSProp AG-1 Linear',"results_model9_improved_training21_seed3_25trainings_50e_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size128",'limegreen',plot_color_median='darkgreen'))
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-2',"results_model9_improved_training27_seed3_30trainings_50e_mnist__nonlinear_layers_avggrad_RMSprop_batch_size128",colors_colorblind_safe7[6],plot_color_median=colors_colorblind_safe7[6]))
#
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-3-1',"results_model9_improved_training30_seed4_50trainings_50e_mnist__nonlinear_layers_avggrad_RMSprop_batch_size128",colors_colorblind_safe6[5],plot_color_median=colors_colorblind_safe6[5]))
#
# data_to_plot.append(utils.PlotStatistics('Adam',"adam_results_model9_not_improved_training_seed18_15trainings_125e_mnist",colors_colorblind_safe6[3],plot_color_median=colors_colorblind_safe6[3]))
#
# data_to_plot.append(utils.PlotStatistics('Soap',"results_model9_not_improved_training20_seed3_50trainings_50e_mnist_SOAP_betas(0.95,0.95)_batch_size128",colors_colorblind_safe6[0],plot_color_median=colors_colorblind_safe6[0]))
# # data_to_plot.append(utils.PlotStatistics('Soap AG-3-1',"results_model9_improved_training30_seed4_50trainings_50e_mnist__nonlinear_layers_avggrad_SOAP_betas(0.95,0.95)_batch_size128",colors_colorblind_safe6[4],plot_color_median=colors_colorblind_safe6[4]))
# data_to_plot.append(utils.PlotStatistics('Soap ($\\beta_2=0$)',"results_model9_not_improved_training20_seed6_50trainings_50e_mnist_SOAP_betas(0.0,0.95)_batch_size128",colors_colorblind_safe6[4],plot_color_median=colors_colorblind_safe6[4]))
#
# # data_to_plot.append(utils.PlotStatistics('Soap AG-2 Linear',"results_model9_improved_training27_seed3_30trainings_50e_mnist__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'red',plot_color_median='darkred'))
# data_to_plot.append(utils.PlotStatistics('Soap AG-2 ($\\beta_2=0$)',"results_model9_improved_training27_seed3_30trainings_50e_mnist__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'black',plot_color_median='black'))
#
#
#
# accuracy_range={'x':[0,127],'y':[95.75,98.3]}
# train_loss_range={'x':None,'y':[0.0,0.3]}
# test_loss_range={'x':None,'y':[0.07,0.3]}
# #
# which_method_for_ratio_of_all_batches='AG'
# # method1_sample_efficiency_comparison='Soap AG-2'#'RMSProp AG-1'#'Adam'
# method1_sample_efficiency_comparison='Soap ($\\beta_2=0$)'
# # method2_sample_efficiency_comparison='Soap'
# method2_sample_efficiency_comparison='RMSProp'

# #model 12 IMDb
# data_to_plot.append(utils.PlotStatistics('Soap w/o Momentum',"results_model12_not_improved_training20_seed3_50trainings_150e_imdb_SOAP_betas(0.0,0.95)_batch_size128",'darkcyan',plot_color_median='darkblue'))

data_to_plot.append(utils.PlotStatistics('RMSProp',"results_model12_not_improved_training_seed12_30trainings_200e_imdb",colors_colorblind_safe6[1],plot_color_median=colors_colorblind_safe6[1]))
data_to_plot.append(utils.PlotStatistics('RMSProp AG-1',"results_model12_improved_training_iter2_seed14_30trainings_200e_imdb",colors_colorblind_safe6[2],plot_color_median=colors_colorblind_safe6[2]))
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-3-1',"results_model12_improved_training30_seed4_50trainings_150e_imdb__nonlinear_layers_avggrad_RMSprop_batch_size128",colors_colorblind_safe6[5],plot_color_median=colors_colorblind_safe6[5]))


data_to_plot.append(utils.PlotStatistics('Adam',"adam_results_model12_not_improved_training_seed19_15trainings_200e_imdb",colors_colorblind_safe6[3],plot_color_median=colors_colorblind_safe6[3]))
#
# # data_to_plot.append(utils.PlotStatistics('RMSProp AG-1 Linear',"results_model12_improved_training21_seed3_25trainings_150e_imdb__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size128",'limegreen',plot_color_median='darkgreen'))
##data_to_plot.append(utils.PlotStatistics('RMSProp AG-2',"results_model12_improved_training27_seed3_10trainings_150e_imdb__nonlinear_layers_avggrad_RMSprop_batch_size128",'orange',plot_color_median='orange'))
# # # data_to_plot.append(utils.PlotStatistics('m22 Soap AG-2 Linear',"results_model12_improved_training22_seed3_50trainings_150e_imdb__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'red',plot_color_median='darkred'))
# # # data_to_plot.append(utils.PlotStatistics('m22 Soap AG-2',"results_model12_improved_training22_seed3_50trainings_150e_imdb__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'black',plot_color_median='black'))
# # data_to_plot.append(utils.PlotStatistics('Soap AG-2 Linear',"results_model12_improved_training27_seed3_10trainings_150e_imdb__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'red',plot_color_median='darkred'))
# # data_to_plot.append(utils.PlotStatistics('Soap AG-2',"results_model12_improved_training27_seed3_10trainings_150e_imdb__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size128",'black',plot_color_median='black'))
#
data_to_plot.append(utils.PlotStatistics('Soap',"results_model12_not_improved_training20_seed3_50trainings_150e_imdb_SOAP_betas(0.95,0.95)_batch_size128",colors_colorblind_safe6[0],plot_color_median=colors_colorblind_safe6[0]))

data_to_plot.append(utils.PlotStatistics('Soap AG-3-1',"results_model12_improved_training30_seed4_50trainings_150e_imdb__nonlinear_layers_avggrad_SOAP_betas(0.95,0.95)_batch_size128",colors_colorblind_safe6[5],plot_color_median=colors_colorblind_safe6[5]))

accuracy_range={'x':None,'y':None}
train_loss_range={'x':None,'y':[0.379,0.7]}
test_loss_range={'x':None,'y':[0.43,0.7]}

which_method_for_ratio_of_all_batches='AG'
# method1_sample_efficiency_comparison='RMSProp'
method1_sample_efficiency_comparison='Adam'
method2_sample_efficiency_comparison='RMSProp AG-1'


# # # #model 13
# data_to_plot.append(utils.PlotStatistics('Soap w/o Momentum',"results_model13_not_improved_training20_seed3_50trainings_10e_imagenet_ood_SOAP_betas(0.0,0.95)_batch_size64",'darkcyan',plot_color_median='darkblue'))
# data_to_plot.append(utils.PlotStatistics('Soap',"results_model13_not_improved_training20_seed3_50trainings_10e_imagenet_ood_SOAP_betas(0.95,0.95)_batch_size64",'magenta',plot_color_median='darkmagenta'))
#
# data_to_plot.append(utils.PlotStatistics('RMSProp',"results_model13_not_improved_training20_seed8_200trainings_10e_imagenet_ood_RMSprop_batch_size64",'brown',plot_color_median='brown'))
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-1',"results_model13_improved_training21_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64",'purple',plot_color_median='purple'))
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-1 Linear',"results_model13_improved_training21_seed3_50trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size64",'limegreen',plot_color_median='darkgreen'))
#
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-2',"results_model13_improved_training27_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64",'orange',plot_color_median='orange'))
# data_to_plot.append(utils.PlotStatistics('RMSProp AG-2 Linear',"results_model13_improved_training27_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size64",'yellow',plot_color_median='yellow'))
#
#
# data_to_plot.append(utils.PlotStatistics('Adam',"results_model13_not_improved_training20_seed8_200trainings_10e_imagenet_ood_Adam_batch_size64",'aquamarine',plot_color_median='aquamarine'))
# data_to_plot.append(utils.PlotStatistics('Adam AG-2',"results_model13_improved_training27_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad_Adam_betas(0.9,0.999)_batch_size64",'purple',plot_color_median='purple'))
# data_to_plot.append(utils.PlotStatistics('Adam AG-2 Linear',"results_model13_improved_training27_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_Adam_betas(0.9,0.999)_batch_size64",'gray',plot_color_median='gray'))
#
# data_to_plot.append(utils.PlotStatistics('Soap AG-1',"results_model13_improved_training21_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size64",'pink',plot_color_median='pink'))
# data_to_plot.append(utils.PlotStatistics('Soap AG-1 Linear',"results_model13_improved_training21_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size64",'darkkhaki',plot_color_median='darkkhaki'))
#
# data_to_plot.append(utils.PlotStatistics('Soap AG-2',"results_model13_improved_training27_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size64",'black',plot_color_median='black'))
# data_to_plot.append(utils.PlotStatistics('Soap AG-2 Linear',"results_model13_improved_training27_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size64",'red',plot_color_median='darkred'))
#
#
#
# which_method_for_ratio_of_all_batches='AG'
# method1_sample_efficiency_comparison='RMSProp'
# # method1_sample_efficiency_comparison='Adam'
# method2_sample_efficiency_comparison='RMSProp AG-1'

# # # #model 18
# # # data_to_plot.append(utils.PlotStatistics('Soap w/o Momentum',"",'darkcyan',plot_color_median='darkblue'))
# # # data_to_plot.append(utils.PlotStatistics('Soap',"",'magenta',plot_color_median='darkmagenta'))
# #
# # data_to_plot.append(utils.PlotStatistics('RMSProp',"results_model18_not_improved_training20_seed3_40trainings_70e_imagenet_ood_RMSprop_batch_size64",'brown',plot_color_median='brown'))
# # data_to_plot.append(utils.PlotStatistics('RMSProp AG-1',"results_model18_improved_training21_seed3_20trainings_70e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64",'purple',plot_color_median='purple'))
# #
# # # data_to_plot.append(utils.PlotStatistics('Adam',"",'aquamarine',plot_color_median='aquamarine'))
# # # data_to_plot.append(utils.PlotStatistics('Adam AG-2',"",'purple',plot_color_median='purple'))
# # # data_to_plot.append(utils.PlotStatistics('Adam AG-2 Linear',"",'gray',plot_color_median='gray'))
# # #
# #
# # # data_to_plot.append(utils.PlotStatistics('RMSProp AG-1 Linear',"",'limegreen',plot_color_median='darkgreen'))
# # data_to_plot.append(utils.PlotStatistics('RMSProp AG-2',"results_model18_improved_training27_seed3_3trainings_70e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64",'orange',plot_color_median='orange'))
# #
# # data_to_plot.append(utils.PlotStatistics('RMSProp AG-2 Linear',"results_model18_improved_training27_seed3_3trainings_70e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size64",'yellow',plot_color_median='yellow'))
# #
# # data_to_plot.append(utils.PlotStatistics('Soap AG-2',"results_model18_improved_training27_seed3_3trainings_70e_imagenet_ood__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size64",'black',plot_color_median='black'))
# # data_to_plot.append(utils.PlotStatistics('Soap AG-2 Linear',"results_model18_improved_training27_seed3_3trainings_70e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size64",'red',plot_color_median='red'))
#
#
#
# data_to_plot.append(utils.PlotStatistics('Soap',"results_model18_not_improved_training20_seed4_20trainings_70e_imagenet_ood_SOAP_betas(0.95,0.95)_batch_size64",'pink',plot_color_median='pink'))
# data_to_plot.append(utils.PlotStatistics('Soap AG',"results_model18_improved_training30_seed4_15trainings_70e_imagenet_ood__nonlinear_layers_avggrad_SOAP_betas(0.95,0.95)_batch_size64",'darkkhaki',plot_color_median='darkkhaki'))
#
# which_method_for_ratio_of_all_batches='AG'
# method1_sample_efficiency_comparison=data_to_plot[1].method_name#'Soap w/o Momentum'
# # method1_sample_efficiency_comparison='Adam'
# method2_sample_efficiency_comparison=data_to_plot[0].method_name#'Soap w/o Momentum AG'


# # which_method_for_ratio_of_all_batches='AG'
# # method1_sample_efficiency_comparison='Soap AG-2 Linear'
# # # method1_sample_efficiency_comparison='Adam'
# # method2_sample_efficiency_comparison='RMSProp'

# # # #model 13, different algorithm
# data_to_plot.append(utils.PlotStatistics('RMSProp',"results_model13_not_improved_training20_seed8_200trainings_10e_imagenet_ood_RMSprop_batch_size64",'brown',plot_color_median='brown'))
# ### data_to_plot.append(utils.PlotStatistics('RMSProp AG-1',"results_model13_improved_training21_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64",'purple',plot_color_median='purple'))
# ### data_to_plot.append(utils.PlotStatistics('RMSProp AG-1 Linear',"results_model13_improved_training21_seed3_50trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size64",'limegreen',plot_color_median='darkgreen'))
#
# ### data_to_plot.append(utils.PlotStatistics('RMSProp AG-2',"results_model13_improved_training27_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64",'orange',plot_color_median='orange'))
# ### data_to_plot.append(utils.PlotStatistics('RMSProp AG-2 Linear',"results_model13_improved_training27_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size64",'yellow',plot_color_median='yellow'))
#
#
# data_to_plot.append(utils.PlotStatistics('Adam',"results_model13_not_improved_training20_seed8_200trainings_10e_imagenet_ood_Adam_batch_size64",'aquamarine',plot_color_median='aquamarine'))
# # # ### data_to_plot.append(utils.PlotStatistics('Adam AG-2',"results_model13_improved_training27_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad_Adam_betas(0.9,0.999)_batch_size64",'purple',plot_color_median='purple'))
# # # ### data_to_plot.append(utils.PlotStatistics('Adam AG-2 Linear',"results_model13_improved_training27_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_Adam_betas(0.9,0.999)_batch_size64",'gray',plot_color_median='gray'))
# # #
# # # ### data_to_plot.append(utils.PlotStatistics('Soap AG-1',"results_model13_improved_training21_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size64",'pink',plot_color_median='pink'))
# # # ### data_to_plot.append(utils.PlotStatistics('Soap AG-1 Linear',"results_model13_improved_training21_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size64",'darkkhaki',plot_color_median='darkkhaki'))
# # #
# # # ### data_to_plot.append(utils.PlotStatistics('Soap AG-2',"results_model13_improved_training27_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size64",'black',plot_color_median='black'))
# # # ### data_to_plot.append(utils.PlotStatistics('Soap AG-2 Linear',"results_model13_improved_training27_seed3_20trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size64",'red',plot_color_median='darkred'))
# # #
# # data_to_plot.append(utils.PlotStatistics('Soap ($\\beta_2=0$)',"results_model13_not_improved_training20_seed3_50trainings_10e_imagenet_ood_SOAP_betas(0.0,0.95)_batch_size64",'blue',plot_color_median='blue'))
# # # data_to_plot.append(utils.PlotStatistics('Soap',"results_model13_not_improved_training20_seed3_50trainings_10e_imagenet_ood_SOAP_betas(0.95,0.95)_batch_size64",'magenta',plot_color_median='darkmagenta'))
# # #
# # #
# # # data_to_plot.append(utils.PlotStatistics('Soap AG-3',"results_model13_improved_training37_seed6_75trainings_10e_imagenet_ood__nonlinear_layers_avggrad_SOAP_betas(0.95,0.95)_batch_size64",'red',plot_color_median='darkred'))
# # data_to_plot.append(utils.PlotStatistics('Soap AG-3 ($\\beta_2=0$)',"results_model13_improved_training37_seed7_75trainings_10e_imagenet_ood__nonlinear_layers_avggrad_SOAP_betas(0.0,0.95)_batch_size64",'purple',plot_color_median='purple'))
# # # # ##### data_to_plot.append(utils.PlotStatistics('Adam AG',"results_model13_improved_training37_seed5_100trainings_10e_imagenet_ood__nonlinear_layers_avggrad_Adam_betas(0.9,0.999)_batch_size64",'pink',plot_color_median='pink'))
# # #
# # # # data_to_plot.append(utils.PlotStatistics('RMSProp AG',"results_model13_improved_training37_seed4_100trainings_10e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64",'black',plot_color_median='black'))
# # # # #
# # # # data_to_plot.append(utils.PlotStatistics('Adam AG',"results_model13_improved_training37_seed7_100trainings_10e_imagenet_ood__nonlinear_layers_avggrad_Adam_betas(0.9,0.999)_batch_size64",'pink',plot_color_median='pink'))
# # #
# # #
# data_to_plot.append(utils.PlotStatistics('Soap (Freq=1)',"freq_preconfitioning_results_model13_not_improved_training20_seed9_75trainings_10e_imagenet_ood_SOAP_betas(0.95,0.95)_batch_size64",'yellow',plot_color_median='yellow'))
#
# data_to_plot.append(utils.PlotStatistics('Soap AG-3 (Freq=1)',"freq_preconfitioning_results_model13_improved_training37_seed8_50trainings_10e_imagenet_ood__nonlinear_layers_avggrad_SOAP_betas(0.95,0.95)_batch_size64",'orange',plot_color_median='orange'))
# #
# #
# # data_to_plot.append(utils.PlotStatistics('RMSProp AG-3',"results_model13_improved_training37_seed4_100trainings_10e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64",'black',plot_color_median='black'))
# # #
# # data_to_plot.append(utils.PlotStatistics('Adam AG-3',"results_model13_improved_training37_seed7_100trainings_10e_imagenet_ood__nonlinear_layers_avggrad_Adam_betas(0.9,0.999)_batch_size64",'pink',plot_color_median='pink'))
# #
# # # data_to_plot.append(utils.PlotStatistics('RMSProp AG-3-A',"results_model13_improved_training33_seed5_100trainings_10e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64",'black',plot_color_median='black'))
# # # #
# # # data_to_plot.append(utils.PlotStatistics('Adam AG-3-A',"results_model13_improved_training33_seed5_100trainings_10e_imagenet_ood__nonlinear_layers_avggrad_Adam_betas(0.9,0.999)_batch_size64",'pink',plot_color_median='pink'))
#
# which_method_for_ratio_of_all_batches='AG'
# # method1_sample_efficiency_comparison='RMSProp'
# # method2_sample_efficiency_comparison='RMSProp AG-3'
# method1_sample_efficiency_comparison=data_to_plot[1].method_name#'Soap w/o Momentum'
# # method1_sample_efficiency_comparison='Adam'
# method2_sample_efficiency_comparison=data_to_plot[0].method_name#'Soap w/o Momentum AG'
#
#
# method_names=[d.method_name for d in data_to_plot]
#
# if 'Soap (Freq=1)' in method_names and 'Soap AG-3 (Freq=1)' in method_names:
#     accuracy_range={'x':[2.7,10],'y':[51.5,53.8]}
# elif 'Soap ($\\beta_2=0$)' in method_names and 'Soap AG-3 ($\\beta_2=0$)' in method_names:
#     accuracy_range={'x':[2.7,10],'y':[51.5,54]}
# elif 'Soap' in method_names and 'Soap AG-3' in method_names:
#     accuracy_range={'x':[2.7,10],'y':[51.5,53.6]}
# elif 'RMSProp' in method_names and 'Adam' in method_names:
#     accuracy_range={'x':[3.7,10],'y':[51.5,52.6]}
# utils.set_automatic_colors(data_to_plot,colors_colorblind_safe6)



for d in data_to_plot:
    d.data_dict=utils.read_file(directory+d.file_name+'.txt')

if plt.rcParams['text.usetex']:
    replace_strings=[('-',r'\raisebox{0.2ex}{-}'),('Freq=1',r'$\text{Freq}=1$')]
    for data in data_to_plot:
        # data.method_name=data.method_name.replace('-1',r'$\text{-}1$')
        # data.method_name = data.method_name.replace('-2', r'$\text{-}2$')
        # data.method_name = data.method_name.replace('-3', r'$\text{-}3$')
        # data.method_name = data.method_name.replace('-3-1', r'$\text{-}3\text{-}1$')
        # data.method_name=data.method_name.replace('-',r'\raisebox{0.2ex}{-}')
        for (s1,s2) in replace_strings:
            data.method_name=data.method_name.replace(s1,s2)
    # method1_sample_efficiency_comparison=method1_sample_efficiency_comparison.replace('-',r'\raisebox{0.2ex}{-}')
    # method2_sample_efficiency_comparison=method2_sample_efficiency_comparison.replace('-',r'\raisebox{0.2ex}{-}')
    for (s1, s2) in replace_strings:
        method1_sample_efficiency_comparison=method1_sample_efficiency_comparison.replace(s1, s2)
        method2_sample_efficiency_comparison=method2_sample_efficiency_comparison.replace(s1, s2)


    # method1_sample_efficiency_comparison=data_to_plot[1].method_name#'Soap w/o Momentum'
    # # method1_sample_efficiency_comparison='Adam'
    # method2_sample_efficiency_comparison=data_to_plot[0].method_name#'Soap w/o Momentum AG'


def count_of_trainings(file_name):
    return int(file_name[(file_name[:file_name.index('trainings_')].rindex('_')+1):file_name.index('trainings_')])
# count_not_improved=count_of_trainings(not_improved_file_name)
# count_improved=count_of_trainings(improved_file_name)
# count_improved_more_iter=count_of_trainings(improved_more_iter_file_name)

for d in data_to_plot:
    d.training_count=count_of_trainings(d.file_name)
    d.count_str=' of '+str(d.file_name)

# count_not_improved=' of '+str(count_not_improved)
# count_improved=' of '+str(count_improved)
# count_improved_more_iter=' of '+str(count_improved_more_iter)

def change_legend_linewidth(width=legend_linewidth):
    leg=plt.legend()
    leg_lines=leg.get_lines()
    #plt.setp(leg_lines,linewidth=width)
    for line in leg_lines:
        if line.get_linewidth()<legend_linewidth:
            line.set_linewidth(legend_linewidth)
        else:
            line.set_linewidth(legend_linewidth*2./1.3333)

figure_num=1
def plot_fig(figure_num=figure_num,y_label=None,metric='train_loss',mean=True,mean_errors=True,median=False,median_errors=False,log_scale=log_scale,ylim=None,xlim=None,data_to_plot=data_to_plot,save_plots=save_plots):
    #global figure_num

    mean_median_settings=[]
    if mean:
        mean_median_settings.append((True,mean_errors))
    if median:
        mean_median_settings.append((False,median_errors))

    for setting in mean_median_settings:
        mean=setting[0]
        err_enabled=setting[1]
        for d in data_to_plot:
            metric_data=d.data_dict[metric]
            means_or_medians,sem_errors=(utils.means_and_sem_err if mean else utils.medians_and_sem_err)(metric_data)
            label=d.get_label('mean' if mean else 'median')
            linestyle=means_linestyle if mean else medians_linestyle
            if d.linestyle is not None:
                linestyle=d.linestyle

            plt.figure(figure_num,**args)
            if log_scale:
                plt.yscale("log")
            # else:
            #     plt.ylim([0, None])


            if err_enabled:
                plt.fill_between(x=means_or_medians.keys(),
                                 y1=[y - utils.confidence_z_score * e for y, e in
                                     zip(means_or_medians.values(), sem_errors.values())],
                                 y2=[y + utils.confidence_z_score * e for y, e in
                                     zip(means_or_medians.values(), sem_errors.values())], alpha=.2, color=d.error_color if mean else d.plot_color_median)
            plt.plot(means_or_medians.keys(), means_or_medians.values(), color=d.plot_color if mean else d.plot_color_median,
                     label=label, linestyle=linestyle,
                     **plot_means_args if mean else {})

    plt.xlabel("Epoch")
    if y_label is None:
        y_label=metric
    plt.ylabel(y_label)
    plt.legend()
    # plt.ylim([1/30, 40])
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)


    change_legend_linewidth()
    # if not log_scale:
    #     plt.gca().ticklabel_format(style='plain', axis='y')
    #figure_num+=1
    if save_plots:
        plt.savefig(save_directory + metric+'_' + data_to_plot[0].file_name + "." + save_format,
                    bbox_inches='tight',
                    dpi=saved_dpi, format=save_format)


try:
    plot_fig(0,'Training Loss','train_loss',mean=plot_means,mean_errors=plot_means_err,median=plot_medians,median_errors=plot_medians_err,log_scale=log_scale,xlim=train_loss_range['x'],ylim=train_loss_range['y'])
except:
    pass
# plt.figure(1,**args)
# not_improved_to_plot=not_improved['train_loss']
# improved_to_plot=improved['train_loss']
# improved_more_iter_to_plot=improved_more_iter['train_loss'] if improved_more_iter is not None else None
# improved_means,improved_sem=utils.means_and_sem_err(improved_to_plot)
# not_improved_means,not_improved_sem=utils.means_and_sem_err(not_improved_to_plot)
# improved_more_iter_means,improved_more_iter_sem=utils.means_and_sem_err(improved_more_iter_to_plot) if improved_more_iter is not None else (None,None)
#
# if log_scale:
#     plt.yscale("log")
# else:
#     plt.ylim([0, None])
#
#
#
# if plot_means:
#     if plot_means_err:
#         plt.fill_between(x=not_improved_means.keys(),
#                          y1=[y - utils.confidence_z_score*e for y, e in zip(not_improved_means.values(), not_improved_sem.values())],
#                         y2=[y + utils.confidence_z_score*e for y, e in zip(not_improved_means.values(), not_improved_sem.values())], alpha=.2,color='darkcyan')
#         plt.fill_between(x=improved_means.keys(),
#                          y1=[y - utils.confidence_z_score*e for y, e in zip(improved_means.values(), improved_sem.values())],
#                         y2=[y + utils.confidence_z_score*e for y, e in zip(improved_means.values(), improved_sem.values())], alpha=.2,color='magenta')
#
#     plt.plot(not_improved_means.keys(),not_improved_means.values(),color='darkcyan',label='Standard Training (Mean'+count_not_improved+')',linestyle=means_linestyle,**plot_means_args)
#     plt.plot(improved_means.keys(),improved_means.values(),color='magenta',label=name+' (2 Iterations; Mean'+count_improved+')',linestyle=means_linestyle,**plot_means_args)
#     if improved_more_iter is not None:
#         if plot_means_err:
#             plt.fill_between(x=improved_more_iter_means.keys(),
#                              y1=[y - utils.confidence_z_score * e for y, e in
#                                  zip(improved_more_iter_means.values(), improved_more_iter_sem.values())],
#                              y2=[y + utils.confidence_z_score * e for y, e in
#                                  zip(improved_more_iter_means.values(), improved_more_iter_sem.values())], alpha=.2,color='limegreen')
#         plt.plot(improved_more_iter_means.keys(), improved_more_iter_means.values(), color='limegreen',
#                  label=name+' (5 Iterations; Mean'+count_improved_more_iter+')',linestyle=means_linestyle,**plot_means_args)
#
#
def medians_plot(_plt=plt,confidence_ranges=True):
    improved_medians, improved_median_sem = utils.medians_and_sem_err(improved_to_plot)
    not_improved_medians, not_improved_median_sem = utils.medians_and_sem_err(not_improved_to_plot)
    improved_more_iter_medians, improved_median_more_iter_sem = utils.medians_and_sem_err(
        improved_more_iter_to_plot) if improved_more_iter is not None else (None, None)


    if len(not_improved_medians.keys())!=0:
        _plt.plot(not_improved_medians.keys(), not_improved_medians.values(), color='darkblue', label='Standard Training (Median'+count_not_improved+')', linestyle=medians_linestyle)
    _plt.plot(improved_medians.keys(), improved_medians.values(), color='darkmagenta', label=name+' (2 Iterations; Median'+count_improved+')', linestyle=medians_linestyle)
    ylim=0
    if improved_more_iter is not None:
        _plt.plot(improved_more_iter_medians.keys(), improved_more_iter_medians.values(), color='darkgreen',
                 label=name+' (5 Iterations; Median'+count_improved_more_iter+')', linestyle=medians_linestyle)
        if plot_medians_err and confidence_ranges and improved_more_iter_medians:
            ylim = _plt.ylim()
            plt.autoscale(False)
            _plt.fill_between(x=improved_more_iter_medians.keys(),
                             y1=[y - utils.confidence_z_score*utils.standard_error_of_the_median_mul * e for y, e in
                                 zip(improved_more_iter_medians.values(), improved_median_more_iter_sem.values())],
                             y2=[y + utils.confidence_z_score*utils.standard_error_of_the_median_mul * e for y, e in
                                 zip(improved_more_iter_medians.values(), improved_median_more_iter_sem.values())], alpha=.2,
                             color='limegreen')

    # plt.yscale("log")
    # plt.draw()
    # xlim=_plt.xlim() if plt==_plt else _plt.get_xlim()
    # ylim = _plt.ylim() if plt == _plt else _plt.get_ylim()
    # if plt == _plt:
    #     _plt.xlim(xlim)
    #     _plt.ylim(ylim)
    #     # _plt.gca().autoscale(tight=True)
    # else:
    #     _plt.set_xlim(xlim)
    #     _plt.set_ylim(ylim)
    #     # _plt.autoscale(tight=True)

    #_plt.gca().set_xlim(_plt.xlim())
    if plot_medians_err and confidence_ranges and not_improved_medians:
        _plt.fill_between(x=not_improved_medians.keys(),
                         y1=[y - utils.confidence_z_score *utils.standard_error_of_the_median_mul* e for y, e in
                             zip(not_improved_medians.values(), not_improved_median_sem.values())],
                         y2=[y + utils.confidence_z_score*utils.standard_error_of_the_median_mul * e for y, e in
                             zip(not_improved_medians.values(), not_improved_median_sem.values())], alpha=.2, color='darkblue')



    if plot_medians_err and confidence_ranges and improved_medians:
        _plt.fill_between(x=improved_medians.keys(),
                         y1=[y - utils.confidence_z_score*utils.standard_error_of_the_median_mul * e for y, e in
                             zip(improved_medians.values(), improved_median_sem.values())],
                         y2=[y + utils.confidence_z_score*utils.standard_error_of_the_median_mul * e for y, e in
                             zip(improved_medians.values(), improved_median_sem.values())], alpha=.2, color='darkmagenta')
        # _plt.draw()
    try:
        if ylim:
            _plt.ylim(ylim)
    except:
        pass
    # if plt == _plt:
    #     if _plt.ylim<ymin
    #     _plt.ylim((ymin,_plt.ylim))
    #     # _plt.gca().autoscale(tight=True)
    # else:
    #     _plt.set_xlim(xlim)
    #     _plt.set_ylim(ylim)
#
# if plot_medians:
#     medians_plot()
#
# #plt.title("Genetic Algorithm Mean Squared Error on Training Dataset for Sorting Algorithm Search")
# plt.xlabel("Epoch")
# plt.ylabel("Training Loss")
# plt.legend()
# #plt.ylim([1/30, 40])
# change_legend_linewidth()
#
# if save_plots:
#     plt.savefig(save_directory + 'train_loss_' + not_improved_file_name + "." + save_format, bbox_inches='tight',
#                 dpi=saved_dpi, format=save_format)
# #plt.show()



# if epoch_ratio_bar_plot:
#     width = 0.8  # the width of the bars: can also be len(x) sequence
#
#     fig, ax = plt.subplots(**args)
#     bottom = np.zeros(len(improved['train_higher_loss_batch_ratio'].keys()),dtype='float64')
#
#     #for sex, sex_count in sex_counts.items():
#     p = ax.bar(improved['train_lower_loss_batch_ratio'].keys(), improved['train_lower_loss_batch_ratio_aggregated'].values(), width, label="Lower Loss Batches\nCompared to Standard Update", bottom=bottom,color='g')
#     bottom += np.array(list(improved['train_lower_loss_batch_ratio_aggregated'].values()))
#     p = ax.bar(improved['train_same_loss_batch_ratio'].keys(), improved['train_same_loss_batch_ratio_aggregated'].values(), width, label="Same Loss Batches", bottom=bottom,color='black')
#     bottom += np.array(list(improved['train_same_loss_batch_ratio_aggregated'].values()))
#     p = ax.bar(improved['train_higher_loss_batch_ratio'].keys(), improved['train_higher_loss_batch_ratio_aggregated'].values(), width, label="Higher Loss Batches", bottom=bottom,color='red')
#
#     #ax.bar_label(p, label_type='center')
#
#     #ax.set_title('Ratios of Batches With Loss Improvement Over Standard Backpropagation Loss Update')
#     ax.legend()
#
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Ratio of All Batches")
#     #plt.show()
# else:
#     plt.figure(2, **args)
#     for (key,color,label) in [('train_lower_loss_batch_ratio','limegreen','Loss Decrease'),('train_same_loss_batch_ratio','black','Same Loss'),('train_higher_loss_batch_ratio','red','Loss Increase')]:
#         #X=list(improved['train_lower_loss_batch_ratio'].keys())
#
#         improved_to_plot = improved[key]
#         improved_more_iter_to_plot = improved_more_iter[key] if improved_more_iter is not None else None
#         improved_means, improved_sem = utils.means_and_sem_err(improved_to_plot)
#         improved_more_iter_means, improved_more_iter_sem = utils.means_and_sem_err(improved_more_iter_to_plot) if improved_more_iter is not None else (None,None)
#
#         if improved_more_iter is not None:
#             plt.fill_between(x=improved_more_iter_means.keys(),
#                          y1=[y - utils.confidence_z_score * e for y, e in
#                              zip(improved_more_iter_means.values(), improved_more_iter_sem.values())],
#                          y2=[y + utils.confidence_z_score * e for y, e in
#                              zip(improved_more_iter_means.values(), improved_more_iter_sem.values())], alpha=.15,color=color)
#         plt.fill_between(x=improved_means.keys(),
#                          y1=[y - utils.confidence_z_score * e for y, e in
#                              zip(improved_means.values(), improved_sem.values())],
#                          y2=[y + utils.confidence_z_score * e for y, e in
#                              zip(improved_means.values(), improved_sem.values())], alpha=.15,color=color)
#         if improved_more_iter is not None:
#             plt.plot(improved_more_iter_means.keys(), improved_more_iter_means.values(), color=color, linestyle="--",
#                  label=label+' (5 Iterations)')
#         plt.plot(improved_means.keys(), improved_means.values(), color=color, linestyle='-',
#                  label=label+' (2 Iterations)')
#
#         # plt.title("Genetic Algorithm Mean Squared Error on Training Dataset for Sorting Algorithm Search")
#         plt.xlabel("Epoch")
#         plt.ylabel("Ratio of All Batches")
#         plt.legend()
#         plt.ylim([0, None])#plt.ylim([0, 1])
#         #plt.ylim([0, 1])
#
# if save_plots:
#     plt.savefig(save_directory + 'train_batch_ratios_' + not_improved_file_name + "." + save_format, bbox_inches='tight',
#                 dpi=saved_dpi, format=save_format)
_data_for_batch_ratios=[copy.deepcopy(d) for d in data_to_plot if 'train_lower_loss_batch_ratio' in d.data_dict and which_method_for_ratio_of_all_batches in d.method_name]
if _data_for_batch_ratios:
    data2=[_data_for_batch_ratios[0]]
    for d in data2:
        d.plot_color=d.error_color='green'
        d.error_color='green'
        d.plot_color='green'
    plot_fig(-1,'Ratio of All Batches','train_lower_loss_batch_ratio',mean=True,mean_errors=True,median=False,median_errors=False,log_scale=False,ylim=None,data_to_plot=data2,save_plots=False)
    for d in data2:
        d.plot_color=d.error_color='black'
    plot_fig(-1,'Ratio of All Batches','train_same_loss_batch_ratio',mean=True,mean_errors=True,median=False,median_errors=False,log_scale=False,ylim=None,data_to_plot=data2,save_plots=False)
    for d in data2:
        d.plot_color=d.error_color='red'
    plot_fig(-1,'Ratio of All Batches','train_higher_loss_batch_ratio',mean=True,mean_errors=True,median=False,median_errors=False,log_scale=False,ylim=None,data_to_plot=data2,save_plots=True)



    data3=[copy.deepcopy(d) for d in data_to_plot if 'train_batch_avg_loss_improvement' in d.data_dict]
    try:
        log_scale_dloss = True
        plot_fig(-2,'$-\\Delta$Loss','train_batch_avg_relative_loss_improvement',mean=True,mean_errors=True,median=False,median_errors=False,log_scale=log_scale_dloss,ylim=None,data_to_plot=data3)
        plot_fig(-2,'$-\\Delta$Loss','train_batch_avg_relative_loss_improvement',mean=True,mean_errors=True,median=False,median_errors=False,log_scale=log_scale_dloss,ylim=None,data_to_plot=data3)
    except:
        log_scale_dloss = False
        plot_fig(-2,'$-\\Delta$Loss','train_batch_avg_relative_loss_improvement',mean=True,mean_errors=True,median=False,median_errors=False,log_scale=log_scale_dloss,ylim=None,data_to_plot=data3)
        plot_fig(-2,'$-\\Delta$Loss','train_batch_avg_relative_loss_improvement',mean=True,mean_errors=True,median=False,median_errors=False,log_scale=log_scale_dloss,ylim=None,data_to_plot=data3)

# try:
#     log_scale_dloss = True
#     plot_fig(-3,'−ΔLoss','train_batch_avg_relative_loss_improvement',mean=True,mean_errors=True,median=True,median_errors=False,log_scale=log_scale_dloss,ylim=None,data_to_plot=data3)
# except:
#     log_scale_dloss = False
#     plot_fig(-3,'−ΔLoss','train_batch_avg_relative_loss_improvement',mean=True,mean_errors=True,median=True,median_errors=False,log_scale=log_scale_dloss,ylim=None,data_to_plot=data3)




# improved_to_plot=improved['train_batch_avg_loss_improvement']
# improved_more_iter_to_plot=improved_more_iter['train_batch_avg_loss_improvement'] if improved_more_iter is not None else None
# improved_means,improved_sem=utils.means_and_sem_err(improved_to_plot)
# improved_more_iter_means,improved_more_iter_sem=utils.means_and_sem_err(improved_more_iter_to_plot) if improved_more_iter is not None else (None,None)
#
#
# plot_double=len(np.where(np.array(list(improved_means.values()))<=0)[0])!=0
# if improved_more_iter is not None and not plot_double:
#     plot_double = len(np.where(np.array(list(improved_more_iter_means.values())) <= 0)[0]) != 0
#
# fig, axs=plt.subplots(2 if plot_double else 1,**args)
# if not plot_double:
#     axs=(axs,)
# # plt.fill_between(x=improved_means.keys(),
# #                  y1=[y - utils.confidence_z_score*e for y, e in zip(improved_means.values(), improved_sem.values())],
# #                 y2=[y + utils.confidence_z_score*e for y, e in zip(improved_means.values(), improved_sem.values())], alpha=.25,color='magenta')
# #plt.plot(improved_means.keys(),improved_means.values(),color='magenta',label='Optimized Training With Optimal Hyperparameters For Standard Training')
# if improved_more_iter is not None:
#     axs[0].errorbar(improved_more_iter_means.keys(),improved_more_iter_means.values(),yerr=utils.confidence_z_score*np.array(list(improved_more_iter_sem.values())),color='limegreen', ecolor=(0.7,1,0.7),label='5 Iterations',linestyle=means_linestyle)
# axs[0].errorbar(improved_means.keys(),improved_means.values(),yerr=utils.confidence_z_score*np.array(list(improved_sem.values())),color='magenta', ecolor=(1,0.7,1),label='2 Iterations',linestyle=means_linestyle)
#
# if plot_medians:
#     not_improved_to_plot={}
#     medians_plot(axs[0],confidence_ranges=False)
#
# #if log_scale:
# #axs[0].yscale("log")
# axs[0].set_yscale("log")
# axs[0].set_xlabel("Epoch")
# axs[0].set_ylabel("−ΔLoss")
# axs[0].legend()
# #plt.show()
#
#
# #plt.figure(4,**args)
#
# if plot_double:
#     #improved_to_plot=improved['train_batch_avg_loss_improvement']
#     #improved_means,improved_sem=utils.means_and_sem_err(improved_to_plot)
#
#     # axs[1].fill_between(x=np.array(list(improved_means.keys()),dtype='float'),
#     #                  y1=np.array([y - utils.confidence_z_score*e for y, e in zip(improved_means.values(), improved_sem.values())],dtype='float'),
#     #                 y2=np.array([y + utils.confidence_z_score*e for y, e in zip(improved_means.values(), improved_sem.values())],dtype='float'), alpha=.25,color='green')
#     #axs[1].plot(improved_means.keys(),improved_means.values(),color='green',label='Optimized Training With Optimal Hyperparameters For Standard Training')
#     r=delta_loss_default_range_linear_scale
#     if improved_more_iter is not None:
#         axs[1].errorbar(list(improved_more_iter_means.keys())[r[0]:r[1]], list(improved_more_iter_means.values())[r[0]:r[1]],
#                         yerr=utils.confidence_z_score * np.array(list(improved_more_iter_sem.values()))[r[0]:r[1]], color='limegreen', ecolor=(0.7,1,0.7),
#                         label='5 Iterations',linestyle=means_linestyle)
#     axs[1].errorbar(list(improved_means.keys())[r[0]:r[1]],list(improved_means.values())[r[0]:r[1]],yerr=utils.confidence_z_score*np.array(list(improved_sem.values()))[r[0]:r[1]],color='magenta', ecolor=(1,0.7,1),label='2 Iterations',linestyle=means_linestyle)
#
#
#     #plt.errorbar(improved_means.keys(),improved_means.values(),yerr=utils.confidence_z_score*np.array(list(improved_sem.values())),color='magenta',label='Optimized Training With Optimal Hyperparameters For Standard Training')
#     if plot_medians:
#         medians_plot(axs[1])
#
#     axs[1].set_xlabel("Epoch")
#     axs[1].set_ylabel("−ΔLoss")
#     axs[1].legend()
#     axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
#     fig.subplots_adjust(hspace=0.25)
#
#     # if log_scale:
#     #     plt.yscale("log")
# if save_plots:
#     plt.savefig(save_directory + 'train_batch_avg_loss_improvement_' + not_improved_file_name + "." + save_format, bbox_inches='tight',
#                 dpi=saved_dpi, format=save_format)
#
#
# improved_to_plot=improved['train_batch_avg_relative_loss_improvement']
# improved_more_iter_to_plot=improved_more_iter['train_batch_avg_relative_loss_improvement'] if improved_more_iter is not None else None
# improved_means,improved_sem=utils.means_and_sem_err(improved_to_plot)
# improved_more_iter_means,improved_more_iter_sem=utils.means_and_sem_err(improved_more_iter_to_plot) if improved_more_iter is not None else (None,None)
#
#
# plot_double=len(np.where(np.array(list(improved_means.values()))<=0)[0])!=0
# if improved_more_iter is not None and not plot_double:
#     plot_double = len(np.where(np.array(list(improved_more_iter_means.values())) <= 0)[0]) != 0
#
#
# #fig, axs=plt.subplots(2 if plot_double else 1,**args)
# #if not plot_double:
# #    axs=(axs,)
# if plot_relative_loss_improvements:
#     plt.figure(10)
#     axs=plt.gca()
#     # plt.fill_between(x=improved_means.keys(),
#     #                  y1=[y - utils.confidence_z_score*e for y, e in zip(improved_means.values(), improved_sem.values())],
#     #                 y2=[y + utils.confidence_z_score*e for y, e in zip(improved_means.values(), improved_sem.values())], alpha=.25,color='magenta')
#     #plt.plot(improved_means.keys(),improved_means.values(),color='magenta',label='Optimized Training With Optimal Hyperparameters For Standard Training')
#     axs.errorbar(improved_means.keys(),improved_means.values(),yerr=utils.confidence_z_score*np.array(list(improved_sem.values())),color='magenta', ecolor=(1,0.7,1),label='2 Iterations',linestyle=means_linestyle)
#     if improved_more_iter is not None:
#         axs.errorbar(improved_more_iter_means.keys(), improved_more_iter_means.values(),
#                      yerr=utils.confidence_z_score * np.array(list(improved_more_iter_sem.values())), color='limegreen',
#                      ecolor=(0.7, 1, 0.7), label='5 Iterations',linestyle=means_linestyle)
#
#     #if log_scale:
#     #axs[0].yscale("log")
#     #axs.set_yscale("log")
#     axs.set_xlabel("Epoch")
#     axs.set_ylabel("−ΔLoss")
#     axs.legend()
#
#     if plot_medians:
#         not_improved_to_plot = {}
#         medians_plot(axs,confidence_ranges=False)
#
#     if save_plots:
#         plt.savefig(save_directory + 'train_batch_avg_relative_loss_improvement_' + not_improved_file_name + "." + save_format, bbox_inches='tight',
#                     dpi=saved_dpi, format=save_format)

for data in data_to_plot:
    print(data.method_name+' - params: '+str(data.data_dict['params']))
print()



for data in data_to_plot:
    print(data.method_name+' - training count: '+str(len(data.data_dict['test_loss_min']))+' ('+str(len(data.data_dict['test_loss'].keys()))+' epochs)')
print()

# print("Standard training count: "+str(len(not_improved['train_loss_min'])))
# print("Improved training count: "+str(len(improved['train_loss_min'])))
# if improved_more_iter is not None:
#     print("Improved training count (more iter): "+str(len(improved_more_iter['train_loss_min'])))
# print()


for data in data_to_plot:
    if 'relative_step_length_l1' in data.data_dict:
        mean,sem=utils.mean_and_sem_err(sum(list(data.data_dict['relative_step_length_l1'].values()),[]))
        print(data.method_name+' - relative step length L1: '+str(mean)+" +/- "+str(sem*utils.confidence_z_score))
        mean, sem = utils.mean_and_sem_err(sum(list(data.data_dict['relative_step_length_l2'].values()),[]))
        print(data.method_name + ' - relative step length L2: ' + str(mean) + " +/- " + str(sem * utils.confidence_z_score))
print()

try:
    for data in data_to_plot:
        mean,sem=utils.mean_and_sem_err(data.data_dict['train_loss_min'])
        print(data.method_name+' - average lowest loss over training: '+str(mean)+" +/- "+str(sem*utils.confidence_z_score))
        means = list(utils.means_and_sem_err(data.data_dict['train_loss'])[0].values())
        argmin = np.argmin(means) + 1
        print("Epoch with best average: " + str(argmin))
    print()
except:
    pass

# improved_mean,improved_sem=utils.mean_and_sem_err(improved['train_loss_min'])
# not_improved_mean,not_improved_sem=utils.mean_and_sem_err(not_improved['train_loss_min'])
# improved_more_iter_means,improved_more_iter_sem=utils.mean_and_sem_err(improved_more_iter['train_loss_min']) if improved_more_iter is not None else (None,None)
#
# if improved_more_iter is not None:
#     print("Improved training average best loss over training (more iter): "+str(improved_more_iter_means)+" +/- "+str(improved_more_iter_sem*utils.confidence_z_score))
#     means=list(utils.means_and_sem_err(improved_more_iter['train_loss'])[0].values())
#     argmin=np.argmin(means)+1
#     print("Epoch with best average: "+str(argmin))
#
# print("Improved training average best loss over training: "+str(improved_mean)+" +/- "+str(improved_sem*utils.confidence_z_score))
# means=list(utils.means_and_sem_err(improved['train_loss'])[0].values())
# argmin=np.argmin(means)+1
# print("Epoch with best average: "+str(argmin))
#
# print("Not improved training average best loss over training: "+str(not_improved_mean)+" +/- "+str(not_improved_sem*utils.confidence_z_score))
# means=list(utils.means_and_sem_err(not_improved['train_loss'])[0].values())
# argmin=np.argmin(means)+1
# print("Epoch with best average: "+str(argmin))
#
# print()

try:
    for data in data_to_plot:
        # mean,sem=utils.mean_and_sem_err(data.data_dict['train_loss_min'])
        # print(data.method_name+' - average lowest loss over training: '+str(mean)+" +/- "+str(sem*utils.confidence_z_score))
        mean,sem = utils.mean_and_sem_err(data.data_dict['train_loss'][list(data.data_dict['train_loss'].keys())[len(data.data_dict['train_loss'].keys())-1]])
        # argmin = np.argmin(means) + 1
        print(data.method_name+ " - last epoch loss: " + str(mean)+' +/- '+str(sem*utils.confidence_z_score))
    print()
except:
    pass
# if improved_more_iter is not None:
#     improved_more_iter_mean, improved_more_iter_sem = utils.mean_and_sem_err(
#         improved_more_iter['train_loss'][list(improved_more_iter['train_loss'].keys())[len(improved_more_iter['train_loss'].keys())-1]])
#     print("Improved training last epoch loss (more iter): "+str(improved_more_iter_mean)+" +/- "+str(improved_more_iter_sem*utils.confidence_z_score))
# improved_mean, improved_sem = utils.mean_and_sem_err(
#     improved['train_loss'][list(improved['train_loss'].keys())[len(improved['train_loss'].keys())-1]])
# print("Improved training last epoch loss: "+str(improved_mean)+" +/- "+str(improved_sem*utils.confidence_z_score))
# not_improved_mean, not_improved_sem = utils.mean_and_sem_err(
#     not_improved['train_loss'][list(not_improved['train_loss'].keys())[len(not_improved['train_loss'].keys())-1]])
# print("Not improved training last epoch loss: "+str(not_improved_mean)+" +/- "+str(not_improved_sem*utils.confidence_z_score))
#
# print()


for data in data_to_plot:
    mean,sem = utils.mean_and_sem_err(data.data_dict['test_loss_min'])
    print(data.method_name+ " - average best test loss over training: " + str(mean)+' +/- '+str(sem*utils.confidence_z_score))
    means = list(utils.means_and_sem_err(data.data_dict['test_loss'])[0].values())
    argmin = np.argmin(means) + 1
    print("Epoch with best average: " + str(argmin))
print()

# improved_mean,improved_sem=utils.mean_and_sem_err(improved['test_loss_min'])
# not_improved_mean,not_improved_sem=utils.mean_and_sem_err(not_improved['test_loss_min'])
# improved_more_iter_means,improved_more_iter_sem=utils.mean_and_sem_err(improved_more_iter['test_loss_min']) if improved_more_iter is not None else (None,None)
#
# if improved_more_iter is not None:
#     print("Improved test average best loss over training (more iter): "+str(improved_more_iter_means)+" +/- "+str(improved_more_iter_sem*utils.confidence_z_score))
#     means=list(utils.means_and_sem_err(improved_more_iter['test_loss'])[0].values())
#     argmin=np.argmin(means)+1
#     print("Epoch with best average: "+str(argmin))
#
# print("Improved test average best loss over training: "+str(improved_mean)+" +/- "+str(improved_sem*utils.confidence_z_score))
# means=list(utils.means_and_sem_err(improved['test_loss'])[0].values())
# argmin=np.argmin(means)+1
# print("Epoch with best average: "+str(argmin))
#
# print("Not improved test average best loss over training: "+str(not_improved_mean)+" +/- "+str(not_improved_sem*utils.confidence_z_score))
# means=list(utils.means_and_sem_err(not_improved['test_loss'])[0].values())
# argmin=np.argmin(means)+1
# print("Epoch with best average: "+str(argmin))
#
# print()

for data in data_to_plot:
    mean,sem = utils.mean_and_sem_err(data.data_dict['test_accuracy_max'])
    print(data.method_name+ " - average best test accuracy over training: " + str(mean)+' +/- '+str(sem*utils.confidence_z_score))
    means = list(utils.means_and_sem_err(data.data_dict['test_accuracy'])[0].values())
    argmax = np.argmax(means) + 1
    print("Epoch with best average: " + str(argmax))
print()

# improved_mean,improved_sem=utils.mean_and_sem_err(improved['test_accuracy_max'])
# not_improved_mean,not_improved_sem=utils.mean_and_sem_err(not_improved['test_accuracy_max'])
# improved_more_iter_means,improved_more_iter_sem=utils.mean_and_sem_err(improved_more_iter['test_accuracy_max']) if improved_more_iter is not None else (None,None)
#
# if improved_more_iter is not None:
#     print("Improved test average best accuracy over training (more iter): "+str(improved_more_iter_means)+" +/- "+str(improved_more_iter_sem*utils.confidence_z_score))
#     means=list(utils.means_and_sem_err(improved_more_iter['test_accuracy'])[0].values())
#     argmax=np.argmax(means)+1
#     print("Epoch with best average: "+str(argmax))
#
# print("Improved test average best accuracy over training: "+str(improved_mean)+" +/- "+str(improved_sem*utils.confidence_z_score))
# means=list(utils.means_and_sem_err(improved['test_accuracy'])[0].values())
# argmax=np.argmax(means)+1
# print("Epoch with best average: "+str(argmax))
#
# print("Not improved test average best accuracy over training: "+str(not_improved_mean)+" +/- "+str(not_improved_sem*utils.confidence_z_score))
# means=list(utils.means_and_sem_err(not_improved['test_accuracy'])[0].values())
# argmax=np.argmax(means)+1
# print("Epoch with best average: "+str(argmax))
# print()

if train_avg_relative_loss_improvement_how_many_outliers_to_exclude!=0:
    key='train_avg_relative_loss_improvement'
    for data in data_to_plot:
        if key in data.data_dict:
            n=train_avg_relative_loss_improvement_how_many_outliers_to_exclude
            x=data.data_dict[key][list(data.data_dict[key].keys())[len(data.data_dict[key].keys())-1]]
            x=sorted(x,reverse=True)
            for _ in range(n):
                x.pop(0)
            mean,sem=utils.mean_and_sem_err(x)
            print(data.method_name+" - average relative loss improvement without "+str(n)+" high outliers: "+ str(mean)+' +/- '+str(sem*utils.confidence_z_score))
    print()


key='train_avg_relative_loss_improvement'
for data in data_to_plot:
    if key in data.data_dict:
        mean,sem=utils.mean_and_sem_err(data.data_dict[key][list(data.data_dict[key].keys())[len(data.data_dict[key].keys())-1]])
        print(data.method_name+" - average relative loss improvement: "+ str(mean)+' +/- '+str(sem*utils.confidence_z_score))
print()

# key='train_avg_relative_loss_improvement'
# if key in improved:
#     if improved_more_iter is not None:
#         improved_more_iter_mean, improved_more_iter_sem = utils.mean_and_sem_err(
#             improved_more_iter[key][list(improved_more_iter[key].keys())[len(improved_more_iter[key].keys())-1]])
#         print("Improved training (more iter) average relative loss improvement: "+str(improved_more_iter_mean)+" +/- "+str(improved_more_iter_sem*utils.confidence_z_score))
#     improved_mean, improved_sem = utils.mean_and_sem_err(
#         improved[key][list(improved[key].keys())[len(improved[key].keys())-1]])
#     print("Improved training average relative loss improvement: "+str(improved_mean)+" +/- "+str(improved_sem*utils.confidence_z_score))
#     print()


# def display_param(param,percentile=None):
#     improved_mean, improved_sem = utils.mean_and_sem_err(list(improved[param].values()))
#     #not_improved_mean, not_improved_sem = utils.mean_and_sem_err(list(not_improved[param].values()))
#     improved_more_iter_means, improved_more_iter_sem = utils.mean_and_sem_err(
#         list(improved_more_iter[param].values())) if improved_more_iter is not None else (None, None)
#     if percentile is not None:
#         improved_mean, improved_sem=(np.percentile(list(improved[param].values()),percentile),float("inf"))
#         improved_more_iter_means, improved_more_iter_sem=(np.percentile(list(improved_more_iter[param].values()),percentile),float("inf")) if improved_more_iter is not None else (None, None)
#         print("Percentile "+str(percentile)+":")
#     if improved_more_iter is not None:
#         print("Improved "+param+" (more iter): " + str(
#             improved_more_iter_means) + " +/- " + str(improved_more_iter_sem * utils.confidence_z_score))
#
#     print("Improved "+param+": " + str(improved_mean) + " +/- " + str(
#         improved_sem * utils.confidence_z_score))
#
#     # print("Not improved "+param+": " + str(not_improved_mean) + " +/- " + str(
#     #     not_improved_sem * utils.confidence_z_score))
#     print()
def display_param_trainingwisely(param,to_percent=True,percentile=None):
    for method_data in data_to_plot:
        data=method_data.data_dict
        if param in data:
            mul=1
            if to_percent:
                mul=100
            vals=[]
            # data=improved
            for t in range(0, len(data[param][1])):
                avg = 0.
                count = 0
                for key in data[param].keys():
                    avg += data[param][key][t]
                    count += 1
                avg /= count
                vals.append(avg)
            improved_mean, improved_sem = utils.mean_and_sem_err(vals)
            if percentile is not None:
                improved_mean, improved_sem=(np.percentile(vals,percentile),1.2533141373155002512078826424055*improved_sem)#float("inf"))
                print("Percentile "+str(percentile)+":")
            print(method_data.method_name+" - " + param + ": " + str(improved_mean*mul) + " +/- " + str(
                improved_sem * utils.confidence_z_score*mul))
            # if improved_more_iter is not None:
            #     data = improved_more_iter
            #     vals=[]
            #     for t in range(0, len(data[param][1])):
            #         avg=0.
            #         count=0
            #         for key in data[param].keys():
            #             avg+=data[param][key][t]
            #             count+=1
            #         avg/=count
            #         vals.append(avg)
            #     improved_more_iter_means, improved_more_iter_sem = utils.mean_and_sem_err(
            #         vals)
            #     if percentile is not None:
            #         improved_more_iter_means, improved_more_iter_sem = (
            #         np.percentile(vals, percentile), 1.2533141373155002512078826424055*improved_more_iter_sem)#float("inf"))
            #     print("Improved " + param + " (more iter): " + str(
            #         improved_more_iter_means*mul) + " +/- " + str(improved_more_iter_sem * utils.confidence_z_score*mul))


display_param_trainingwisely('train_higher_loss_batch_ratio')
display_param_trainingwisely('train_same_loss_batch_ratio')
display_param_trainingwisely('train_lower_loss_batch_ratio')
# display_param('train_higher_loss_batch_ratio_aggregated')
# display_param('train_same_loss_batch_ratio_aggregated')
# display_param('train_lower_loss_batch_ratio_aggregated')

# display_param('train_batch_avg_loss_improvement_aggregated',30)
display_param_trainingwisely('train_batch_avg_loss_improvement',percentile=50,to_percent=False)
# print(sorted(improved["train_loss"][18],reverse=True))
# print(sorted(improved["train_loss"][29],reverse=True))
# print(sorted(not_improved["train_loss"][29],reverse=True))
# print(sorted(improved["train_loss"][30],reverse=True))


#print(stats.normaltest(not_improved['test_loss'][28]))
#print(stats.normaltest(not_improved['test_loss'][29]))


def test_param(param,p_value=0.05,verbose=False):
    verbose_lvl=2 if verbose else 1

    try:
        print('Tests for ' + param + ':')
        for data in data_to_plot:
            if param in data.data_dict:
                print(data.method_name+': ',end='')
                normality_test(data.data_dict[param],p_value,verbose_lvl)
    except:
        pass
    print('--------------------------------------')
    # try:
    #     print('Tests for '+param+':')
    #     if param in not_improved:
    #         print('not_improved: ',end='')
    #         normality_test(not_improved[param],p_value,verbose_lvl)
    #     print('improved: ', end='')
    #     normality_test(improved[param],p_value,verbose_lvl)
    #     if improved_more_iter is not None:
    #         print('improved_more_iter: ', end='')
    #         normality_test(improved_more_iter[param],p_value,verbose_lvl)
    # except:
    #     pass
    # print('--------------------------------------')


# normality_test(not_improved['test_loss'])
# normality_test(improved['test_loss'])
# normality_test(improved_more_iter['test_loss'])
# #normality_test(improved['test_loss'])
# normality_test(not_improved['train_loss'])
# normality_test(improved['train_loss'])
# normality_test(improved_more_iter['train_loss'])
#
# normality_test(not_improved['test_accuracy'])
# normality_test(improved['test_loss'])
# normality_test(improved_more_iter['test_loss'])
if test_normality:
    try:
        print()
        test_param('test_loss',verbose=False)
        test_param('train_loss',verbose=False)
        test_param('test_accuracy',verbose=False)#test_param('test_accuracy',verbose=True)
        test_param('train_batch_avg_loss_improvement',verbose=False)
        test_param('train_lower_loss_batch_ratio',verbose=False)
        test_param('train_same_loss_batch_ratio',verbose=False)
        test_param('train_higher_loss_batch_ratio',verbose=False)
        print()
        print('-------------------------------------------------------------------')
        print()
        test_param('test_loss',p_value=0.5,verbose=False)
        test_param('train_loss',p_value=0.5,verbose=False)
        test_param('test_accuracy',p_value=0.5,verbose=False)#test_param('test_accuracy',verbose=True)
        test_param('train_batch_avg_loss_improvement',p_value=0.5,verbose=False)
        test_param('train_lower_loss_batch_ratio',p_value=0.5,verbose=False)
        test_param('train_same_loss_batch_ratio',p_value=0.5,verbose=False)
        test_param('train_higher_loss_batch_ratio',p_value=0.5,verbose=False)
    except:
        print('Exception occurred in normality tests')
    print()
    print('-------------------------------------------------------------------')
    print()
    def last_epoch_stats(param):
        for method_data in data_to_plot:
            pval=float('inf')
            data=method_data.data_dict
            if param in data:
                pval = utils.single_normality_test(data[param][len(data[param]) - 1])
                print(method_data.method_name+' - ' + 'Last epoch '+param+' p-val: ' + str(pval))

        # not_improved_pval=float('inf')
        # improved_pval = float('inf')
        # improved_more_iter_pval = float('inf')
        # data=not_improved
        # if param in data:
        #     not_improved_pval=utils.single_normality_test(data[param][len(data[param])-1])
        #     print('not improved - '+'Last epoch test loss p-val: '+str(not_improved_pval))
        # data = improved
        # if param in data:
        #     improved_pval = utils.single_normality_test(data[param][len(data[param]) - 1])
        #     print('improved - ' + 'Last epoch test loss p-val: ' + str(improved_pval))
        # data = improved_more_iter
        # if data is not None and param in data:
        #     improved_more_iter_pval = utils.single_normality_test(data[param][len(data[param]) - 1])
        #     print('improved more iter - ' + 'Last epoch test loss p-val: ' + str(improved_more_iter_pval))
        # return (not_improved_pval,improved_pval,improved_more_iter_pval)

    try:
        last_epoch_stats('test_loss')
    except:
        print('Exception occurred in normality tests')


mean_sample_efficiency=0.
switch_method_mean_sample_efficiency=False
if method1_sample_efficiency_comparison in [data.method_name for data in data_to_plot] and method2_sample_efficiency_comparison in [data.method_name for data in data_to_plot]:
    data1 = [data for data in data_to_plot if data.method_name == method1_sample_efficiency_comparison][0]
    data2 = [data for data in data_to_plot if data.method_name == method2_sample_efficiency_comparison][0]
    # values1=(list(data1.data_dict[sample_efficiency_comparison_metric].values()))
    # values2=(list(data2.data_dict[sample_efficiency_comparison_metric].values()))
    values1, _ = utils.means_and_sem_err(data1.data_dict[sample_efficiency_comparison_metric])
    values2, _ = utils.means_and_sem_err(data2.data_dict[sample_efficiency_comparison_metric])
    values1 = list(values1.values())
    values2 = list(values2.values())
    switch=np.min(values1)<np.min(values2)
    _method1_sample_efficiency_comparison, _method2_sample_efficiency_comparison=method1_sample_efficiency_comparison,method2_sample_efficiency_comparison
    if switch:
        values1,values2=values2,values1
        _method1_sample_efficiency_comparison,_method2_sample_efficiency_comparison=method2_sample_efficiency_comparison,method1_sample_efficiency_comparison
    switch_method_mean_sample_efficiency=switch

    min_of_worse_method=np.min(values1)
    arg_min_of_worse_method=np.argmin(values1)
    ind=0
    arg_matching_of_better_method=-2
    for ind in range(len(values2)):
        if values2[ind]<min_of_worse_method:
            arg_matching_of_better_method=(abs(values2[ind]-min_of_worse_method)*(ind-1)+abs(values2[ind-1]-min_of_worse_method)*(ind))/abs(values2[ind]-values2[ind-1])
            break
    arg_min_of_worse_method+=1
    arg_matching_of_better_method+=1

    # worse_method_name=method2_sample_efficiency_comparison if switch else method1_sample_efficiency_comparison
    # better_method_name=method1_sample_efficiency_comparison if switch else method2_sample_efficiency_comparison
    print()
    mean_sample_efficiency=arg_min_of_worse_method/arg_matching_of_better_method
    print("Sample efficiency gain of "+_method2_sample_efficiency_comparison+" compared to "+_method1_sample_efficiency_comparison+" (mean): "+str(mean_sample_efficiency))


if method1_sample_efficiency_comparison in [data.method_name for data in data_to_plot] and method2_sample_efficiency_comparison in [data.method_name for data in data_to_plot]:
    data1=[data for data in data_to_plot if data.method_name==method1_sample_efficiency_comparison][0]
    data2 = [data for data in data_to_plot if data.method_name == method2_sample_efficiency_comparison][0]
    # values1=(list(data1.data_dict[sample_efficiency_comparison_metric].values()))
    # values2=(list(data2.data_dict[sample_efficiency_comparison_metric].values()))
    values1,_ = utils.medians_and_sem_err(data1.data_dict[sample_efficiency_comparison_metric])
    values2,_ = utils.medians_and_sem_err(data2.data_dict[sample_efficiency_comparison_metric])
    values1=list(values1.values())
    values2=list(values2.values())
    switch=np.min(values1)<np.min(values2)
    if switch:
        values1,values2=values2,values1
        method1_sample_efficiency_comparison,method2_sample_efficiency_comparison=method2_sample_efficiency_comparison,method1_sample_efficiency_comparison

    min_of_worse_method=np.min(values1)
    arg_min_of_worse_method=np.argmin(values1)
    ind=0
    arg_matching_of_better_method=-2
    for ind in range(len(values2)):
        if values2[ind]<min_of_worse_method:
            arg_matching_of_better_method=(abs(values2[ind]-min_of_worse_method)*(ind-1)+abs(values2[ind-1]-min_of_worse_method)*(ind))/abs(values2[ind]-values2[ind-1])
            break
    arg_min_of_worse_method+=1
    arg_matching_of_better_method+=1

    # worse_method_name=method2_sample_efficiency_comparison if switch else method1_sample_efficiency_comparison
    # better_method_name=method1_sample_efficiency_comparison if switch else method2_sample_efficiency_comparison
    #print()
    median_sample_efficiency=arg_min_of_worse_method/arg_matching_of_better_method
    print("Sample efficiency gain of "+method2_sample_efficiency_comparison+" compared to "+method1_sample_efficiency_comparison+" (median): "+str(median_sample_efficiency))
    #print()
    if switch==switch_method_mean_sample_efficiency:
        print(
            "Sample efficiency gain of " + method2_sample_efficiency_comparison + " compared to " + method1_sample_efficiency_comparison + " (geometric avg of mean efficiency and median efficiency): " + str(
                (median_sample_efficiency*mean_sample_efficiency)**0.5))
    else:
        print(
            "Sample efficiency gain of " + method2_sample_efficiency_comparison + " and " + method1_sample_efficiency_comparison + " (geometric avg of mean efficiency and median efficiency): " + str(
                (max(median_sample_efficiency/mean_sample_efficiency,mean_sample_efficiency/median_sample_efficiency)) ** 0.5))

data1=[data for data in data_to_plot if data.method_name==method1_sample_efficiency_comparison][0]
data2 = [data for data in data_to_plot if data.method_name == method2_sample_efficiency_comparison][0]
efficiency=[]

first_iter=True
first_iter_data1_max_ind=[]
first_iter_data2_max_ind=[]

modified_data1_copy=copy.deepcopy(data1.data_dict[sample_efficiency_comparison_metric])
modified_data2_copy=copy.deepcopy(data2.data_dict[sample_efficiency_comparison_metric])

while True:
    # if len(efficiency)==26:
    #     print()

    values1_means, _ = utils.means_and_sem_err(data1.data_dict[sample_efficiency_comparison_metric])
    values2_means, _ = utils.means_and_sem_err(data2.data_dict[sample_efficiency_comparison_metric])
    values1_means = list(values1_means.values())
    values2_means = list(values2_means.values())
    if not values1_means or not values2_means:
        break
    switch_means = np.min(values1_means) < np.min(values2_means)
    # _method1_sample_efficiency_comparison, _method2_sample_efficiency_comparison = method1_sample_efficiency_comparison, method2_sample_efficiency_comparison
    if switch_means:
        values1_means, values2_means = values2_means, values1_means
        # _method1_sample_efficiency_comparison, _method2_sample_efficiency_comparison = method2_sample_efficiency_comparison, method1_sample_efficiency_comparison
    switch_method_mean_sample_efficiency = switch_means

    min_of_worse_method_means = np.min(values1_means)
    arg_min_of_worse_method_means = np.argmin(values1_means)
    ind_means = 0
    arg_matching_of_better_method_means = -2

    finish=False
    for ind_means in range(len(values2_means)):
        if values2_means[ind_means] < min_of_worse_method_means:
            if ind_means==0 or values2_means[ind_means]==values2_means[ind_means - 1]:
                finish=True
                break
            arg_matching_of_better_method_means = (abs(values2_means[ind_means] - min_of_worse_method_means) * (ind_means - 1) + abs(
                values2_means[ind_means - 1] - min_of_worse_method_means) * (ind_means)) / abs(values2_means[ind_means] - values2_means[ind_means - 1])
            break
    if finish:
        break
    arg_min_of_worse_method_means += 1
    arg_matching_of_better_method_means += 1

    if first_iter:
        if not switch_means:
            first_iter_data1_max_ind.append(arg_min_of_worse_method_means)
            first_iter_data2_max_ind.append(arg_matching_of_better_method_means)
        else:
            first_iter_data2_max_ind.append(arg_min_of_worse_method_means)
            first_iter_data1_max_ind.append(arg_matching_of_better_method_means)

    mean_sample_efficiency = arg_min_of_worse_method_means / arg_matching_of_better_method_means

    if switch_means:
        mean_sample_efficiency=1./mean_sample_efficiency
    #     efficiency.append((1./mean_sample_efficiency,1))#,0.5*(len(data1.data_dict[sample_efficiency_comparison_metric].keys())+len(data2.data_dict[sample_efficiency_comparison_metric].keys()))))
    # else:
    #     efficiency.append((mean_sample_efficiency,1))#,0.5*(len(data1.data_dict[sample_efficiency_comparison_metric].keys())+len(data2.data_dict[sample_efficiency_comparison_metric].keys()))))





    values1, _ = utils.medians_and_sem_err(data1.data_dict[sample_efficiency_comparison_metric])
    values2, _ = utils.medians_and_sem_err(data2.data_dict[sample_efficiency_comparison_metric])
    values1 = list(values1.values())
    values2 = list(values2.values())
    switch = np.min(values1) < np.min(values2)
    if switch:
        values1, values2 = values2, values1
        # method1_sample_efficiency_comparison, method2_sample_efficiency_comparison = method2_sample_efficiency_comparison, method1_sample_efficiency_comparison

    min_of_worse_method = np.min(values1)
    arg_min_of_worse_method = np.argmin(values1)
    ind = 0
    arg_matching_of_better_method = -2
    for ind in range(len(values2)):
        if values2[ind] < min_of_worse_method:
            if ind==0 or values2[ind]==values2[ind - 1]:
                finish=True
                break
            arg_matching_of_better_method = (abs(values2[ind] - min_of_worse_method) * (ind - 1) + abs(
                values2[ind - 1] - min_of_worse_method) * (ind)) / abs(values2[ind] - values2[ind - 1])
            break
    if finish:
        break
    arg_min_of_worse_method += 1
    arg_matching_of_better_method += 1

    if first_iter:
        if not switch:
            first_iter_data1_max_ind.append(arg_min_of_worse_method)
            first_iter_data2_max_ind.append(arg_matching_of_better_method)
        else:
            first_iter_data2_max_ind.append(arg_min_of_worse_method)
            first_iter_data1_max_ind.append(arg_matching_of_better_method)

    # worse_method_name=method2_sample_efficiency_comparison if switch else method1_sample_efficiency_comparison
    # better_method_name=method1_sample_efficiency_comparison if switch else method2_sample_efficiency_comparison
    # print()
    median_sample_efficiency = arg_min_of_worse_method / arg_matching_of_better_method

    if switch:
        median_sample_efficiency=1/median_sample_efficiency
    efficiency.append(((mean_sample_efficiency * median_sample_efficiency)**0.5, 1))
    #     #efficiency.append((1./median_sample_efficiency,1))#,0.5*(len(data1.data_dict[sample_efficiency_comparison_metric].keys())+len(data2.data_dict[sample_efficiency_comparison_metric].keys()))))
    # else:
    #     efficiency.append(((mean_sample_efficiency*median_sample_efficiency)**0.5, 1))
    #     #efficiency.append((median_sample_efficiency,1))#0.5*(len(data1.data_dict[sample_efficiency_comparison_metric].keys())+len(data2.data_dict[sample_efficiency_comparison_metric].keys()))))

    #if switch:
    # if (mean_sample_efficiency < 1. and median_sample_efficiency < 1.) or (
    #         mean_sample_efficiency > 1. and median_sample_efficiency > 1.):
    #     data2.data_dict[sample_efficiency_comparison_metric].pop(
    #         sorted(data2.data_dict[sample_efficiency_comparison_metric].keys())[-1])
    #     data1.data_dict[sample_efficiency_comparison_metric].pop(sorted(data1.data_dict[sample_efficiency_comparison_metric].keys())[-1])
    #
    # elif efficiency[len(efficiency)-1][0]<1.:
    #     data1.data_dict[sample_efficiency_comparison_metric].pop(sorted(data1.data_dict[sample_efficiency_comparison_metric].keys())[-1])
    # else:
    #     data2.data_dict[sample_efficiency_comparison_metric].pop(sorted(data2.data_dict[sample_efficiency_comparison_metric].keys())[-1])

    if first_iter:
        first_iter=False
        bound1=np.max(first_iter_data1_max_ind)
        while math.ceil(bound1) in data1.data_dict[sample_efficiency_comparison_metric].keys():
            data1.data_dict[sample_efficiency_comparison_metric].pop(
                sorted(data1.data_dict[sample_efficiency_comparison_metric].keys())[-1])

        bound2=np.max(first_iter_data2_max_ind)
        while math.ceil(bound2) in data2.data_dict[sample_efficiency_comparison_metric].keys():
            data2.data_dict[sample_efficiency_comparison_metric].pop(
                sorted(data2.data_dict[sample_efficiency_comparison_metric].keys())[-1])

    data1_redundant_tail=0
    data2_redundant_tail=0
    if not switch_means:
        data1_redundant_tail+=len(data1.data_dict[sample_efficiency_comparison_metric].keys())-arg_min_of_worse_method_means
        data2_redundant_tail+=len(data2.data_dict[sample_efficiency_comparison_metric].keys())-arg_matching_of_better_method_means
        # first_iter_data1_max_ind.append(arg_min_of_worse_method_means)
        # first_iter_data2_max_ind.append(arg_matching_of_better_method_means)
    else:
        data1_redundant_tail += len(
            data1.data_dict[sample_efficiency_comparison_metric].keys()) - arg_matching_of_better_method_means
        data2_redundant_tail += len(
            data2.data_dict[sample_efficiency_comparison_metric].keys()) - arg_min_of_worse_method_means
        # first_iter_data2_max_ind.append(arg_min_of_worse_method_means)
        # first_iter_data1_max_ind.append(arg_matching_of_better_method_means)

    if not switch:
        data1_redundant_tail+=len(data1.data_dict[sample_efficiency_comparison_metric].keys())-arg_min_of_worse_method
        data2_redundant_tail+=len(data2.data_dict[sample_efficiency_comparison_metric].keys())-arg_matching_of_better_method
        # first_iter_data1_max_ind.append(arg_min_of_worse_method_means)
        # first_iter_data2_max_ind.append(arg_matching_of_better_method_means)
    else:
        data1_redundant_tail += len(
            data1.data_dict[sample_efficiency_comparison_metric].keys()) - arg_matching_of_better_method
        data2_redundant_tail += len(
            data2.data_dict[sample_efficiency_comparison_metric].keys()) - arg_min_of_worse_method
        # first_iter_data2_max_ind.append(arg_min_of_worse_method_means)
        # first_iter_data1_max_ind.append(arg_matching_of_better_method_means)

    if data1_redundant_tail==data2_redundant_tail:
        data2.data_dict[sample_efficiency_comparison_metric].pop(
            sorted(data2.data_dict[sample_efficiency_comparison_metric].keys())[-1])
        data1.data_dict[sample_efficiency_comparison_metric].pop(sorted(data1.data_dict[sample_efficiency_comparison_metric].keys())[-1])

    elif data1_redundant_tail>data2_redundant_tail:
        data1.data_dict[sample_efficiency_comparison_metric].pop(sorted(data1.data_dict[sample_efficiency_comparison_metric].keys())[-1])
    else:
        data2.data_dict[sample_efficiency_comparison_metric].pop(sorted(data2.data_dict[sample_efficiency_comparison_metric].keys())[-1])



sample_efficiency=0
weight_sum=0
for eff,weight in efficiency:
    sample_efficiency+=eff*weight
    weight_sum+=weight
sample_efficiency/=weight_sum

print(
    "Sample efficiency gain of " + method2_sample_efficiency_comparison + " compared to " + method1_sample_efficiency_comparison + " (for many training points via iterative shortening of trainings): " + str(
        sample_efficiency))

data1.data_dict[sample_efficiency_comparison_metric]=modified_data1_copy
data2.data_dict[sample_efficiency_comparison_metric]=modified_data2_copy


efficiency=[]

# first_iter=True
# first_iter_data1_max_ind=[]
# first_iter_data2_max_ind=[]
del first_iter
del first_iter_data1_max_ind
del first_iter_data2_max_ind

modified_data1_copy=copy.deepcopy(data1.data_dict[sample_efficiency_comparison_metric])
modified_data2_copy=copy.deepcopy(data2.data_dict[sample_efficiency_comparison_metric])

first_iter=True
range_length_adjustment=True

while True:
    # if len(efficiency)==26:
    #     print()

    values1_means, _ = utils.means_and_sem_err(data1.data_dict[sample_efficiency_comparison_metric])
    values2_means, _ = utils.means_and_sem_err(data2.data_dict[sample_efficiency_comparison_metric])
    values1_means = list(values1_means.values())
    values2_means = list(values2_means.values())
    if not values1_means or not values2_means:
        break
    switch_means = np.min(values1_means) < np.min(values2_means)
    # _method1_sample_efficiency_comparison, _method2_sample_efficiency_comparison = method1_sample_efficiency_comparison, method2_sample_efficiency_comparison
    if switch_means:
        values1_means, values2_means = values2_means, values1_means
        # _method1_sample_efficiency_comparison, _method2_sample_efficiency_comparison = method2_sample_efficiency_comparison, method1_sample_efficiency_comparison
    switch_method_mean_sample_efficiency = switch_means

    min_of_worse_method_means = np.min(values1_means)
    arg_min_of_worse_method_means = np.argmin(values1_means)
    ind_means = 0
    arg_matching_of_better_method_means = -2

    finish=False
    for ind_means in range(len(values2_means)):
        if values2_means[ind_means] < min_of_worse_method_means:
            if ind_means==0 or values2_means[ind_means]==values2_means[ind_means - 1]:
                finish=True
                break
            arg_matching_of_better_method_means = (abs(values2_means[ind_means] - min_of_worse_method_means) * (ind_means - 1) + abs(
                values2_means[ind_means - 1] - min_of_worse_method_means) * (ind_means)) / abs(values2_means[ind_means] - values2_means[ind_means - 1])
            break
    if finish:
        break
    arg_min_of_worse_method_means += 1
    arg_matching_of_better_method_means += 1

    # if first_iter:
    #     if not switch_means:
    #         first_iter_data1_max_ind.append(arg_min_of_worse_method_means)
    #         first_iter_data2_max_ind.append(arg_matching_of_better_method_means)
    #     else:
    #         first_iter_data2_max_ind.append(arg_min_of_worse_method_means)
    #         first_iter_data1_max_ind.append(arg_matching_of_better_method_means)

    mean_sample_efficiency = arg_min_of_worse_method_means / arg_matching_of_better_method_means

    if switch_means:
        mean_sample_efficiency=1./mean_sample_efficiency
    #     efficiency.append((1./mean_sample_efficiency,1))#,0.5*(len(data1.data_dict[sample_efficiency_comparison_metric].keys())+len(data2.data_dict[sample_efficiency_comparison_metric].keys()))))
    # else:
    #     efficiency.append((mean_sample_efficiency,1))#,0.5*(len(data1.data_dict[sample_efficiency_comparison_metric].keys())+len(data2.data_dict[sample_efficiency_comparison_metric].keys()))))





    values1, _ = utils.medians_and_sem_err(data1.data_dict[sample_efficiency_comparison_metric])
    values2, _ = utils.medians_and_sem_err(data2.data_dict[sample_efficiency_comparison_metric])
    values1 = list(values1.values())
    values2 = list(values2.values())
    switch = np.min(values1) < np.min(values2)
    if switch:
        values1, values2 = values2, values1
        # method1_sample_efficiency_comparison, method2_sample_efficiency_comparison = method2_sample_efficiency_comparison, method1_sample_efficiency_comparison

    min_of_worse_method = np.min(values1)
    arg_min_of_worse_method = np.argmin(values1)
    ind = 0
    arg_matching_of_better_method = -2
    for ind in range(len(values2)):
        if values2[ind] < min_of_worse_method:
            if ind==0 or values2[ind]==values2[ind - 1]:
                finish=True
                break
            arg_matching_of_better_method = (abs(values2[ind] - min_of_worse_method) * (ind - 1) + abs(
                values2[ind - 1] - min_of_worse_method) * (ind)) / abs(values2[ind] - values2[ind - 1])
            break
    if finish:
        break
    arg_min_of_worse_method += 1
    arg_matching_of_better_method += 1

    # if first_iter:
    #     if not switch:
    #         first_iter_data1_max_ind.append(arg_min_of_worse_method)
    #         first_iter_data2_max_ind.append(arg_matching_of_better_method)
    #     else:
    #         first_iter_data2_max_ind.append(arg_min_of_worse_method)
    #         first_iter_data1_max_ind.append(arg_matching_of_better_method)

    # worse_method_name=method2_sample_efficiency_comparison if switch else method1_sample_efficiency_comparison
    # better_method_name=method1_sample_efficiency_comparison if switch else method2_sample_efficiency_comparison
    # print()
    median_sample_efficiency = arg_min_of_worse_method / arg_matching_of_better_method

    if switch:
        median_sample_efficiency=1/median_sample_efficiency
    efficiency.append(((mean_sample_efficiency * median_sample_efficiency)**0.5, 1))
    #     #efficiency.append((1./median_sample_efficiency,1))#,0.5*(len(data1.data_dict[sample_efficiency_comparison_metric].keys())+len(data2.data_dict[sample_efficiency_comparison_metric].keys()))))
    # else:
    #     efficiency.append(((mean_sample_efficiency*median_sample_efficiency)**0.5, 1))
    #     #efficiency.append((median_sample_efficiency,1))#0.5*(len(data1.data_dict[sample_efficiency_comparison_metric].keys())+len(data2.data_dict[sample_efficiency_comparison_metric].keys()))))

    #if switch:
    # if (mean_sample_efficiency < 1. and median_sample_efficiency < 1.) or (
    #         mean_sample_efficiency > 1. and median_sample_efficiency > 1.):
    #     data2.data_dict[sample_efficiency_comparison_metric].pop(
    #         sorted(data2.data_dict[sample_efficiency_comparison_metric].keys())[-1])
    #     data1.data_dict[sample_efficiency_comparison_metric].pop(sorted(data1.data_dict[sample_efficiency_comparison_metric].keys())[-1])
    #
    # elif efficiency[len(efficiency)-1][0]<1.:
    #     data1.data_dict[sample_efficiency_comparison_metric].pop(sorted(data1.data_dict[sample_efficiency_comparison_metric].keys())[-1])
    # else:
    #     data2.data_dict[sample_efficiency_comparison_metric].pop(sorted(data2.data_dict[sample_efficiency_comparison_metric].keys())[-1])

    # if first_iter:
    #     first_iter=False
    #     bound1=np.max(first_iter_data1_max_ind)
    #     while math.ceil(bound1) in data1.data_dict[sample_efficiency_comparison_metric].keys():
    #         data1.data_dict[sample_efficiency_comparison_metric].pop(
    #             sorted(data1.data_dict[sample_efficiency_comparison_metric].keys())[-1])
    #
    #     bound2=np.max(first_iter_data2_max_ind)
    #     while math.ceil(bound2) in data2.data_dict[sample_efficiency_comparison_metric].keys():
    #         data2.data_dict[sample_efficiency_comparison_metric].pop(
    #             sorted(data2.data_dict[sample_efficiency_comparison_metric].keys())[-1])
    if first_iter:
        if range_length_adjustment:
            max_epoch_used=max(arg_min_of_worse_method_means,arg_matching_of_better_method_means,arg_min_of_worse_method,arg_matching_of_better_method)
            keys=list(data1.data_dict[sample_efficiency_comparison_metric].keys())
            for key in keys:
                if key>max_epoch_used:
                    data1.data_dict[sample_efficiency_comparison_metric].pop(key)
            keys=list(data2.data_dict[sample_efficiency_comparison_metric].keys())
            for key in keys:
                if key>max_epoch_used:
                    data2.data_dict[sample_efficiency_comparison_metric].pop(key)
        first_iter=False


    data1_len=len(data1.data_dict[sample_efficiency_comparison_metric].keys())
    data2_len=len(data2.data_dict[sample_efficiency_comparison_metric].keys())
    if data1_len==data2_len:
        data2.data_dict[sample_efficiency_comparison_metric].pop(
            sorted(data2.data_dict[sample_efficiency_comparison_metric].keys())[-1])
        data1.data_dict[sample_efficiency_comparison_metric].pop(
            sorted(data1.data_dict[sample_efficiency_comparison_metric].keys())[-1])

    elif data1_len>data2_len:
        data1.data_dict[sample_efficiency_comparison_metric].pop(
            sorted(data1.data_dict[sample_efficiency_comparison_metric].keys())[-1])
    else:
        data2.data_dict[sample_efficiency_comparison_metric].pop(
            sorted(data2.data_dict[sample_efficiency_comparison_metric].keys())[-1])



    # data1_redundant_tail=0
    # data2_redundant_tail=0
    # if not switch_means:
    #     data1_redundant_tail+=len(data1.data_dict[sample_efficiency_comparison_metric].keys())-arg_min_of_worse_method_means
    #     data2_redundant_tail+=len(data2.data_dict[sample_efficiency_comparison_metric].keys())-arg_matching_of_better_method_means
    #     # first_iter_data1_max_ind.append(arg_min_of_worse_method_means)
    #     # first_iter_data2_max_ind.append(arg_matching_of_better_method_means)
    # else:
    #     data1_redundant_tail += len(
    #         data1.data_dict[sample_efficiency_comparison_metric].keys()) - arg_matching_of_better_method_means
    #     data2_redundant_tail += len(
    #         data2.data_dict[sample_efficiency_comparison_metric].keys()) - arg_min_of_worse_method_means
    #     # first_iter_data2_max_ind.append(arg_min_of_worse_method_means)
    #     # first_iter_data1_max_ind.append(arg_matching_of_better_method_means)
    #
    # if not switch:
    #     data1_redundant_tail+=len(data1.data_dict[sample_efficiency_comparison_metric].keys())-arg_min_of_worse_method
    #     data2_redundant_tail+=len(data2.data_dict[sample_efficiency_comparison_metric].keys())-arg_matching_of_better_method
    #     # first_iter_data1_max_ind.append(arg_min_of_worse_method_means)
    #     # first_iter_data2_max_ind.append(arg_matching_of_better_method_means)
    # else:
    #     data1_redundant_tail += len(
    #         data1.data_dict[sample_efficiency_comparison_metric].keys()) - arg_matching_of_better_method
    #     data2_redundant_tail += len(
    #         data2.data_dict[sample_efficiency_comparison_metric].keys()) - arg_min_of_worse_method
    #     # first_iter_data2_max_ind.append(arg_min_of_worse_method_means)
    #     # first_iter_data1_max_ind.append(arg_matching_of_better_method_means)
    #
    # if data1_redundant_tail==data2_redundant_tail:
    #     data2.data_dict[sample_efficiency_comparison_metric].pop(
    #         sorted(data2.data_dict[sample_efficiency_comparison_metric].keys())[-1])
    #     data1.data_dict[sample_efficiency_comparison_metric].pop(sorted(data1.data_dict[sample_efficiency_comparison_metric].keys())[-1])
    #
    # elif data1_redundant_tail>data2_redundant_tail:
    #     data1.data_dict[sample_efficiency_comparison_metric].pop(sorted(data1.data_dict[sample_efficiency_comparison_metric].keys())[-1])
    # else:
    #     data2.data_dict[sample_efficiency_comparison_metric].pop(sorted(data2.data_dict[sample_efficiency_comparison_metric].keys())[-1])



sample_efficiency=0
weight_sum=0
for eff,weight in efficiency:
    sample_efficiency+=eff*weight
    weight_sum+=weight
sample_efficiency/=weight_sum

print(
    "Sample efficiency gain of " + method2_sample_efficiency_comparison + " compared to " + method1_sample_efficiency_comparison + " (averaged for all epochs): " + str(
        sample_efficiency))
print(
    "Sample efficiency gain of " + method1_sample_efficiency_comparison + " compared to " + method2_sample_efficiency_comparison + " (averaged for all epochs): " + str(
        1./sample_efficiency))

data1.data_dict[sample_efficiency_comparison_metric]=modified_data1_copy
data2.data_dict[sample_efficiency_comparison_metric]=modified_data2_copy

# not_improved_to_plot=not_improved['train_loss']
# improved_to_plot=improved['train_loss']
# improved_more_iter_to_plot=improved_more_iter['train_loss'] if improved_more_iter is not None else None
# improved_means,improved_sem=utils.means_and_sem_err(improved_to_plot)
# not_improved_means,not_improved_sem=utils.means_and_sem_err(not_improved_to_plot)
# improved_more_iter_means,improved_more_iter_sem=utils.means_and_sem_err(improved_more_iter_to_plot) if improved_more_iter is not None else (None,None)
# if improved_more_iter is not None:
#     better_epochs=0
#     epochs=0
#     for i in improved_more_iter_means.keys():
#         epochs+=1
#         if improved_more_iter_means[i]+improved_more_iter_sem[i]<-not_improved_sem[i]+not_improved_means[i]:
#             better_epochs+=1
#         elif improved_means[i]+improved_sem[i]<-not_improved_sem[i]+not_improved_means[i]:
#             better_epochs += 1
#     print('Percentage of epochs where any improved method is significantly better: '+str(100.*better_epochs/epochs)+'  '+str(better_epochs)+'/'+str(epochs))
# if improved_more_iter is not None:
#     better_epochs=0
#     epochs=0
#     for i in improved_more_iter_means.keys():
#         epochs+=1
#         if improved_more_iter_means[i]+improved_more_iter_sem[i]<-not_improved_sem[i]+not_improved_means[i]:
#             better_epochs+=1
#         # elif improved_means[i]+improved_sem[i]<-not_improved_sem[i]+not_improved_means[i]:
#         #     better_epochs += 1
#     print('Percentage of epochs where improved method with more iterations is significantly better: '+str(100.*better_epochs/epochs)+'  '+str(better_epochs)+'/'+str(epochs))
# # if improved_more_iter is not None:
# better_epochs=0
# epochs=0
# for i in improved_more_iter_means.keys():
#     epochs+=1
#     # if improved_more_iter_means[i]+improved_more_iter_sem[i]<-not_improved_sem[i]+not_improved_means[i]:
#     #     better_epochs+=1
#     if improved_means[i]+improved_sem[i]<-not_improved_sem[i]+not_improved_means[i]:
#         better_epochs += 1
# print('Percentage of epochs where improved method is significantly better: '+str(100.*better_epochs/epochs)+'  '+str(better_epochs)+'/'+str(epochs))
# if improved_more_iter is not None:
#     better_epochs=0
#     epochs=0
#     for i in improved_more_iter_means.keys():
#         epochs+=1
#         if improved_more_iter_means[i]-improved_more_iter_sem[i]>not_improved_sem[i]+not_improved_means[i]:
#             better_epochs+=1
#         elif improved_means[i]-improved_sem[i]>not_improved_sem[i]+not_improved_means[i]:
#             better_epochs += 1
#     print('Percentage of epochs where any of both improved method is significantly worse: '+str(100.*better_epochs/epochs)+'  '+str(better_epochs)+'/'+str(epochs))
# if improved_more_iter is not None:
#     better_epochs=0
#     epochs=0
#     for i in improved_more_iter_means.keys():
#         epochs+=1
#         if improved_more_iter_means[i]-improved_more_iter_sem[i]>not_improved_sem[i]+not_improved_means[i]:
#             better_epochs+=1
#         # elif improved_means[i]+improved_sem[i]<-not_improved_sem[i]+not_improved_means[i]:
#         #     better_epochs += 1
#     print('Percentage of epochs where improved method with more iterations is significantly worse: '+str(100.*better_epochs/epochs)+'  '+str(better_epochs)+'/'+str(epochs))
# # if improved_more_iter is not None:
# better_epochs=0
# epochs=0
# for i in improved_more_iter_means.keys():
#     epochs+=1
#     # if improved_more_iter_means[i]+improved_more_iter_sem[i]<-not_improved_sem[i]+not_improved_means[i]:
#     #     better_epochs+=1
#     if improved_means[i]-improved_sem[i]>not_improved_sem[i]+not_improved_means[i]:
#         better_epochs += 1
# print('Percentage of epochs where improved method is significantly worse: '+str(100.*better_epochs/epochs)+'  '+str(better_epochs)+'/'+str(epochs))


# if plot_test_loss:
#     plt.figure(4, **args)
#     not_improved_to_plot = not_improved['test_loss']
#     improved_to_plot = improved['test_loss']
#     improved_more_iter_to_plot = improved_more_iter['test_loss'] if improved_more_iter is not None else None
#     improved_more_iter_means, improved_more_iter_sem = utils.means_and_sem_err(improved_more_iter_to_plot) if improved_more_iter is not None else (None,None)
#     improved_means, improved_sem = utils.means_and_sem_err(improved_to_plot)
#     not_improved_means, not_improved_sem = utils.means_and_sem_err(not_improved_to_plot)
#
#     if plot_means:
#         if plot_means_err:
#             plt.fill_between(x=not_improved_means.keys(),
#                              y1=[y - utils.confidence_z_score * e for y, e in
#                                  zip(not_improved_means.values(), not_improved_sem.values())],
#                              y2=[y + utils.confidence_z_score * e for y, e in
#                                  zip(not_improved_means.values(), not_improved_sem.values())], alpha=.25, color='darkcyan')
#             plt.fill_between(x=improved_means.keys(),
#                              y1=[y - utils.confidence_z_score * e for y, e in
#                                  zip(improved_means.values(), improved_sem.values())],
#                              y2=[y + utils.confidence_z_score * e for y, e in
#                                  zip(improved_means.values(), improved_sem.values())], alpha=.25, color='magenta')
#         plt.plot(not_improved_means.keys(), not_improved_means.values(), color='darkcyan',
#                  label='Standard Training (Mean'+count_not_improved+')',linestyle=means_linestyle,**plot_means_args)
#         plt.plot(improved_means.keys(), improved_means.values(), color='magenta',
#                  label=name+' (2 Iterations; Mean'+count_improved+')',linestyle=means_linestyle,**plot_means_args)
#         if improved_more_iter is not None:
#             if plot_means_err:
#                 plt.fill_between(x=improved_more_iter_means.keys(),
#                              y1=[y - utils.confidence_z_score * e for y, e in
#                                  zip(improved_more_iter_means.values(), improved_more_iter_sem.values())],
#                              y2=[y + utils.confidence_z_score * e for y, e in
#                                  zip(improved_more_iter_means.values(), improved_more_iter_sem.values())], alpha=.25, color='limegreen')
#             plt.plot(improved_more_iter_means.keys(), improved_more_iter_means.values(), color='limegreen',
#                      label=name+' (5 Iterations; Mean'+count_improved_more_iter+')',linestyle=means_linestyle,**plot_means_args)
#
#     if plot_medians:
#         medians_plot(plt)
#
#     plt.xlabel("Epoch")
#     plt.ylabel("Test Loss")
#     plt.legend()
#     change_legend_linewidth()
#
#     if log_scale:
#         plt.yscale("log")
#     else:
#         plt.ylim([0, None])
#     if save_plots:
#         plt.savefig(save_directory + 'test_loss_' + not_improved_file_name + "." + save_format, bbox_inches='tight',
#                 dpi=saved_dpi, format=save_format)
#
# if plot_test_acc:
#     plt.figure(5, **args)
#     not_improved_to_plot = not_improved['test_accuracy']
#     improved_to_plot = improved['test_accuracy']
#     improved_more_iter_to_plot = improved_more_iter['test_accuracy'] if improved_more_iter is not None else None
#     improved_more_iter_means, improved_more_iter_sem = utils.means_and_sem_err(
#         improved_more_iter_to_plot) if improved_more_iter is not None else (None, None)
#     improved_means, improved_sem = utils.means_and_sem_err(improved_to_plot)
#     not_improved_means, not_improved_sem = utils.means_and_sem_err(not_improved_to_plot)
#
#     if plot_means:
#         plt.fill_between(x=not_improved_means.keys(),
#                          y1=[y - utils.confidence_z_score * e for y, e in
#                              zip(not_improved_means.values(), not_improved_sem.values())],
#                          y2=[y + utils.confidence_z_score * e for y, e in
#                              zip(not_improved_means.values(), not_improved_sem.values())], alpha=.25, color='darkcyan')
#         plt.fill_between(x=improved_means.keys(),
#                          y1=[y - utils.confidence_z_score * e for y, e in
#                              zip(improved_means.values(), improved_sem.values())],
#                          y2=[y + utils.confidence_z_score * e for y, e in
#                              zip(improved_means.values(), improved_sem.values())], alpha=.25, color='magenta')
#         plt.plot(not_improved_means.keys(), not_improved_means.values(), color='darkcyan',
#                  label='Standard Training (Mean'+count_not_improved+')',linestyle=means_linestyle,**plot_means_args)
#         plt.plot(improved_means.keys(), improved_means.values(), color='magenta',
#                  label=name+' (2 Iterations; Mean'+count_improved+')',linestyle=means_linestyle,**plot_means_args)
#         if improved_more_iter is not None:
#             plt.fill_between(x=improved_more_iter_means.keys(),
#                              y1=[y - utils.confidence_z_score * e for y, e in
#                                  zip(improved_more_iter_means.values(), improved_more_iter_sem.values())],
#                              y2=[y + utils.confidence_z_score * e for y, e in
#                                  zip(improved_more_iter_means.values(), improved_more_iter_sem.values())], alpha=.25,
#                              color='limegreen')
#             plt.plot(improved_more_iter_means.keys(), improved_more_iter_means.values(), color='limegreen',
#                      label=name+' (5 Iterations; Mean'+count_improved_more_iter+')',linestyle=means_linestyle,**plot_means_args)
#
#     if plot_medians:
#         medians_plot(plt)
#
#
#     plt.xlabel("Epoch")
#     plt.ylabel("Test Accuracy")
#     plt.legend()
#     change_legend_linewidth()
#     #if log_scale:
#     #    plt.yscale("log")
#     #else:
#     #    plt.ylim([0, None])
#     if 'y' in accuracy_range.keys():
#         plt.ylim(accuracy_range['y'])
#     if 'x' in accuracy_range.keys():
#         plt.xlim(accuracy_range['x'])
#     if save_plots:
#         plt.savefig(save_directory+'test_accuracy_'+not_improved_file_name+"."+save_format,bbox_inches='tight',dpi=saved_dpi,format=save_format)


if plot_test_loss:
    figure_num = 20
    #plot_fig('Test Loss','test_loss')
    plot_fig(figure_num,'Test Loss', 'test_loss', mean=plot_means,mean_errors=plot_means_err,median=plot_medians,median_errors=plot_medians_err,
             log_scale=log_scale,xlim=test_loss_range['x'], ylim=test_loss_range['y'])
if plot_test_acc:
    figure_num = 21
    plot_fig(figure_num,'Test Accuracy', 'test_accuracy', mean=plot_means,mean_errors=plot_means_err,median=plot_medians,median_errors=plot_medians_err,
             log_scale=log_scale, ylim=accuracy_range['y'],xlim=accuracy_range['x'])


def min_statistic_for_a_part_of_training(epoch_start,epoch_end,data):
    mins=[]
    for i in range(len(data[epoch_end])):
        min=np.inf
        for j in range(epoch_start,epoch_end+1):
            if data[j][i]<min:
                min=data[j][i]
        mins.append(min)
    return utils.mean_and_sem_err(mins)



# epochs_compute_min=0#150
# data_to_compute_min=improved_more_iter['train_loss']
#
# if epochs_compute_min>0:
#     print()
#     print()
#     min_loss_150,pm_150=min_statistic_for_a_part_of_training(1,epochs_compute_min,data_to_compute_min)
#     print("Min. loss for the first "+str(epochs_compute_min)+" trainings (improved training): "+str(min_loss_150)+' +/- '+str(pm_150))



plt.show()

#plt.savefig('filename.png', dpi=300)

