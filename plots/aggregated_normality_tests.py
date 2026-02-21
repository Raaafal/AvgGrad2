import utils
import json
import numpy as np
import statistics
from matplotlib.ticker import MaxNLocator
from scipy import stats
from utils import normality_test

directory='plot_data/'
improved_more_iter_file_name=None

experiments={'not_improved_high_lr':['results_model5_not_improved_training_seed0_200trainings_30e',
                                     'results_model5_not_improved_training_seed23_200trainings_30e_not_mnist',
                                     'results_model6_not_improved_training_seed23_200trainings_40e',
                                     'results_model6_not_improved_training_seed23_200trainings_40e_not_mnist'],
             'improved_high_lr':['results_model5_improved_training_seed24_100trainings_30e',
                                 'results_model5_improved_training_seed23_100trainings_30e_not_mnist',
                                 'results_model6_improved_training_seed23_100trainings_40e',
                                 'results_model6_improved_training_seed23_100trainings_40e_not_mnist'],
             'improved_more_iter_high_lr':['results_model5_improved_training_iter5_seed23_50trainings_30e',
                                           'results_model5_improved_training_iter5_seed23_50trainings_30e_not_mnist',
                                           'results_model6_improved_training_iter5_seed23_50trainings_40e',
                                           'results_model6_improved_training_iter5_seed23_50trainings_40e_not_mnist']}

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

def test_param(param,data,p_value=0.05,verbose=False):
    verbose_lvl = 0
    if verbose:
        verbose_lvl=1
    if verbose:
        print('Tests for '+param+':')
    percentage=float('inf')
    if param in data:
        if verbose:
            print('not_improved: ',end='')
        percentage=normality_test(data[param],p_value,verbose_lvl=verbose_lvl)
    return percentage

def last_epoch_stats(param,data,verbose=False):
    pval=float('inf')
    #improved_pval = float('inf')
    #improved_more_iter_pval = float('inf')
    #data=not_improved
    if param in data:
        pval=utils.single_normality_test(data[param][len(data[param])-1])
        if verbose:
            print('Last epoch p-val: '+str(pval))
    return pval

query='not_improved_high_lr'#'improved_more_iter_high_lr'
data=experiments[query]
parameters=['train_loss','test_loss']
last_epoch_stats_parameters=['train_loss']

for parameter in parameters:
    print(parameter+' ',end='')
    stats = []
    stats_last_epoch = []

    log_s = ''
    if 'not_improved' in query:
        log_s = 'not_improved'
    elif 'improved_more_iter' in query:
        log_s = 'improved_more_iter'
    elif 'improved' in query:
        log_s = 'improved'
    print(' ' + log_s + ' ', end='')

    for experiment in data:
        #for experiment in experiments[experiment_group]:
        edata = utils.read_file(directory + experiment + '.txt')
        stats.append(test_param(parameter,edata))
        if parameter in last_epoch_stats_parameters:
            stats_last_epoch.append(last_epoch_stats(parameter,edata))
    print('Stat mean: ' + str(utils.mean(stats))+'   geo mean: '+str(utils.geo_mean(stats)))
    if len(stats_last_epoch)!=0:
        print('Last epoch stat mean: ' + str(utils.mean(stats_last_epoch)) + '   last epoch stat geo mean: ' + str(utils.geo_mean(stats_last_epoch)))



# not_improved=utils.read_file(directory+not_improved_file_name+'.txt')
# improved=utils.read_file(directory+improved_file_name+'.txt')
# improved_more_iter=utils.read_file(directory+improved_more_iter_file_name+'.txt') if improved_more_iter_file_name is not None else None

