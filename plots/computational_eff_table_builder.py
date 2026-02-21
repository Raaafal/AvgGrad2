import numpy as np

import utils

directory='data_for_table/'

methods=[('RMSProp','results_model13_not_improved_training20_seed7_3trainings_10e_imagenet_ood_RMSprop_batch_size64',None),
        ('RMSProp AG-1*','results_model13_improved_training21_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64',None),
        ('RMSProp AG-2*','results_model13_improved_training27_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64',None),
        ('RMSProp AG-3','results_model13_improved_training37_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64',None),
        ('RMSProp AG-1 Linear*','results_model13_improved_training21_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size64',None),
        ('RMSProp AG-2 Linear*','results_model13_improved_training27_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size64','\\\\\\hline\\hline'),

        ('Adam','results_model13_not_improved_training20_seed7_3trainings_10e_imagenet_ood_Adam_betas(0.9,0.999)_batch_size64',None),
        ('Adam AG-3','results_model13_improved_training37_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_Adam_betas(0.9,0.999)_batch_size64','\\\\\\hline\\hline'),

        ('Soap','results_model13_not_improved_training20_seed7_3trainings_10e_imagenet_ood_SOAP_batch_size64',None),
        ('Soap AG-2*','results_model13_improved_training27_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_SOAP_batch_size64',None),
        ('Soap AG-2 Linear*','results_model13_improved_training27_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_batch_size64',None),
        ('Soap AG-3','results_model13_improved_training37_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_SOAP_batch_size64','\\\\\\hline\\hline'),

        ('Soap $\\text{Freq}=1$','results_model13_not_improved_training20_seed7_3trainings_10e_imagenet_ood_SOAP_betas(0.95,0.95)_batch_size64',None),
        ('Soap AG-3 $\\text{Freq}=1$','results_model13_improved_training37_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_SOAP_betas(0.95,0.95)_batch_size64',None)
]

# methods=[('RMSProp','results_model13_not_improved_training20_seed7_3trainings_10e_imagenet_ood_RMSprop_batch_size64',None),
#         ('RMSProp AG-1*','results_model13_improved_training21_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64',None),
#         ('RMSProp AG-2*','results_model13_improved_training26_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64',None),
#         ('RMSProp AG-3','results_model13_improved_training37_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_RMSprop_batch_size64',None),
#         ('RMSProp AG-1 Linear*','results_model13_improved_training21_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size64',None),
#         ('RMSProp AG-2 Linear*','results_model13_improved_training26_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_RMSprop_batch_size64','\\\\\\hline\\hline'),
#
#         ('Adam','results_model13_not_improved_training20_seed7_3trainings_10e_imagenet_ood_Adam_betas(0.9,0.999)_batch_size64',None),
#         ('Adam AG-3','results_model13_improved_training37_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_Adam_betas(0.9,0.999)_batch_size64','\\\\\\hline\\hline'),
#
#         ('Soap','results_model13_not_improved_training20_seed7_3trainings_10e_imagenet_ood_SOAP_batch_size64',None),
#         ('Soap AG-2*','results_model13_improved_training26_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_SOAP_batch_size64',None),
#         ('Soap AG-2 Linear*','results_model13_improved_training26_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad__linear_layers_avggrad_SOAP_batch_size64',None),
#         ('Soap AG-3','results_model13_improved_training37_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_SOAP_batch_size64','\\\\\\hline\\hline'),
#
#         ('Soap $\\text{Freq}=1$','results_model13_not_improved_training20_seed7_3trainings_10e_imagenet_ood_SOAP_betas(0.95,0.95)_batch_size64',None),
#         ('Soap AG-3 $\\text{Freq}=1$','results_model13_improved_training37_seed7_3trainings_10e_imagenet_ood__nonlinear_layers_avggrad_SOAP_betas(0.95,0.95)_batch_size64',None)
# ]


default_separator='\\\\\\hline'
decimal_places=(2,2,1)

# def process_data(data):
#     mean, mean_err = utils.mean_and_sem_err(data['cuda_max_reserved_memory_max'])

dataset_size=25432
def handle_data(method,data):
    s=''
    s+=method[0]+'&'
    mean, mean_err = utils.mean_and_sem_err(data['cuda_max_allocated_memory_max'])
    mean_err = np.std(data['cuda_max_allocated_memory_max'])
    mean=round(mean,decimal_places[0])
    mean_err=round(mean_err,decimal_places[0])
    s+=f'${mean}\pm{mean_err}$&'
    mean, mean_err = utils.mean_and_sem_err(data['cuda_max_reserved_memory_max'])
    mean_err = np.std(data['cuda_max_reserved_memory_max'])
    mean = round(mean, decimal_places[1])
    mean_err = round(mean_err, decimal_places[1])
    s += f'${mean}\pm{mean_err}$&'

    times = []
    for train_times in data['train_time'].values():
        times += train_times
    efficiencies=[]
    for time in times:
        efficiencies.append(dataset_size/time)
    mean, mean_err = utils.mean_and_sem_err(efficiencies)
    mean_err = np.std(efficiencies)
    mean = round(mean, decimal_places[2])
    mean_err = round(mean_err, decimal_places[2])
    s += f'${mean}\pm{mean_err}$'


    if method[2]:
        s+=method[2]
    else:
        s+=default_separator
    return s+'\n'

# data = []
for method in methods:
    data_dict=utils.read_file(directory+method[1]+'.txt')
    # process_data(data_dict)
    # data.append(data_dict)
    s=''
    s+=handle_data(method,data_dict)
    print(s)