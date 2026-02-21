from __future__ import print_function

import torch.nn

#from __future__ import annotations
if __name__=='__main__':
    # import argparse
    # import ast
    import os
    import time

    # import torch
    # import torch.nn as nn
    # import torch.nn.functional as F
    import torch.optim as optim
    # from torch.autograd import Variable
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import StepLR
    # import copy
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    import plots.utils
    import imdb_utils
    #import imagenet_ood_utils
    #from imagenet_ood_utils import get_approx_optimal_class_mapping, get_logit_mask, get_dataloaders, load_resnet, convert_resnet_to_module_graph
    from imagenet_ood_utils import switch_activations_in_dependency_graph,switch_RELU_to_SiLU_in_dependency_graph,switch_RELU_to_GELU_in_dependency_graph, get_dataloaders, load_resnet,convert_resnet_to_module_graph,get_approx_optimal_class_mapping,get_logit_mask
    from algorithms import *
    # from algorithms_linear import *
    #from plots.utils import *
    import gc
    from SOAP import SOAP

    #0 is the standard gradient RMSProp, 1 is average gradient with fixed step size, 2 is average gradient with varying step size, 3 is not ready, 4 is average gradient without predicting potential weight change (doesn't work well for that reason), 5 is the average gradient without switching step direction when the average gradient is positive, method 6 is the average gradient with multiplication of bad updates by a constant, method 7 is testing much further update, method 8 is reimplementation of method 1, method 9 is reimplementation of method 1 but with a small enhancement for 3 or more iterations (!!!!!), method 10 combines momentum features of Adam and RMSprop with momentum, method 11 is like method 1, but with momentum (RMSprop with momentum using avg gradient), method 12 is like method 1, but with momentum (Adam using avg gradient), method 20 is like method 0, but supports logit mask and remapping of targets, method 21 is method 1, but with optimized memory, similarly method 22 is method 2 with optimized memory. Method 25 cancels the update of those parameters, where gradient and average gradient directions doesn't match (the same sign). All other parameters are updated according to the gradient. Method 26: like method 22, but the average gradient is normalized separately to match the gradient norm for every tensor separately, 27 is like 26, but update is further normalized according to the manhattan distance.
    #28 refers to first taking a step of an optimizer with momentum with 10x higher learning rate, and then on that update rage there is computed the average gradient, which is plugged into the optimizer, but without momentum
    #29 refers to the same as 28, but there is no 10x multiplier of the first 'candidate' update
    #30: calculating the average gradient on the range of the previous weight update
    #31 is like 30, but calculating 2 regular gradients
    #32 is like 31, but adjusting the update length to match the gradient update for the most recent weights
    #method 33 calculates the average gradient on the range from n%t weight updates to the current weights
    #method 34 is like method 33, but it allows for negations of gradient values only. It seems worse than 33
    #methods 35 and 36 compute the average gradient for gradients between 2 models that are trained concurrently, for now they doesn't work well
    #method 37 is upgraded method 33, where two past parameter sets are stored, and the oldest is taken each time as the range of gradient averaging
    #method 38 is upgraded method 33 and 37, but there is a buffer of model's states on disk, so that the average gradient is computed on approximately equal ranges of past parameter updates.
    #method 41 is like method 27, but it compares loss minimization to step with average of two gradients
    method=37#31#0,1 or 2, or 4-9
    #upgraded_training = True

    optimizer_type=SOAP#optim.RMSprop#SOAP#optim.Adam#SOAP#optim.RMSprop#optim.Adam#optim.RMSprop
    if (method>=1 and method<=11) or (method>=20 and method<=40):
        optimizer_type = optim.RMSprop
        optimizer_type = SOAP
        # optimizer_type=optim.Adam
    elif method==12:
        optimizer_type = optim.Adam

    # average_gradient_of_linear_layers_enhancement=False#for method=1 or method=2
    # average_gradient_of_nonlinear_layers_enhancement=True#for method=1 or method=2
    # average_gradient_of_loss=False#for method=1 and iter_count = 2, seems to not affect

    if method==4:
        average_gradient_of_loss = False
        average_gradient_of_linear_layers_enhancement = False
        average_gradient_of_nonlinear_layers_enhancement = True

    dataset='imagenet_ood'#'mnist','fashion_mnist' or 'imdb' or 'imagenet_ood'
    #mnist=False#gradient_factor should be smaller for mnist=False
    model_nr=13#14#13#12#9#7 #3 or 4, 5, 6
    model_layer_graph=True
    force_memory_cleanup=False
    if dataset=='imdb':
        model_nr=12
    elif dataset=='imagenet_ood':
        #model_nr=13
        model_layer_graph = True
        force_memory_cleanup=True

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')#torch.device("cuda")

    mini_batch=True
    batch_size=128#8192#4096#128#4096#128#128#2048#128#2048#128#2048#60000#512#512#64 #for minibatch=False higher batch_size is numerically better
    if dataset=='imagenet_ood':
        batch_size=64#2#64#64#64
        # if not torch.cuda.is_available():
        #     batch_size=4#64#for non-productive environment

    cross_val=False
    calculate_test_stats=True
    calculate_train_stats=True

    #trainings_same_model means how many trainings of the model with the same hyperparameters, trainings_different_model: how many different sets of hyperparameters are tested
    trainings_same_model = 75# 100 if method!=0 else 200  # 5#5#2#1#30#3 #or number of folds for cross_val=True
    trainings_different_model = 1  # 100#2#100

    separate_hyperparameters_for_optimized=True
    if method==10 or method==11 or method==12:
        separate_hyperparameters_for_optimized=False

    lr_mul=1.#0.0009/0.00025#1.#0.0009/1.178e-4#1.#0.1*0.00025/1.178e-5#1.#3.48*2.19#1.#0.0009/0.000258#0.0009/0.000258#3.3333333333333333333333333
    optim_args={}
    if method==10 or method==11:
        optim_args={'momentum':0.9,'alpha':0.999}
    elif method>=20 and method<=40 and optimizer_type==SOAP:
        #optim_args = {'betas': (0.9, 0.999)}#Adam-like
        #optim_args = {'betas': (0.995, 0.999)}  # Adam-like2
        #optim_args = {'betas': (0., 0.99)}#RMSprop-like
        optim_args = {'betas': (0.95, 0.95)}#default
        #optim_args = {'betas': (0., 0.95)}#RMSprop-like2
        #optim_args = {'betas': (0.95, 0.95),'precondition_frequency':1}#frequent preconditioning
    elif method==0:
        pass
        #optim_args = {'momentum': 0.9, 'alpha': 0.999}
    elif method==12:
        optim_args = {'betas': (0.9,0.999)}
    elif method>=20 and method<=40 and optimizer_type==optim.Adam:
        optim_args = {'betas': (0.9, 0.999)}#Adam-like
        #optim_args = {'betas': (0.995, 0.999)}  # Adam-like2

    # elif method==23 or method==20:
    # #     optim_args = {'betas': (0.99,0.999)}
    #     optim_args = {'betas': (0.995,0.999)}
    if method==32:
        if 'betas' in optim_args:
            betas=optim_args['betas']
            optim_args['betas']=(betas[0]**0.5,betas[1]**0.5)
        lr_mul*=0.5

    show_grad=False

    seed=6
    epochs=500#2500#300#5000
    if model_nr==6:
        epochs=15
    elif model_nr==9:
        # epochs=500
        # if method>=1:
        #     epochs=300
        epochs = 125
        if method >= 1:
            if trainings_different_model > 1:
                epochs = 50
            else:
                epochs = 50
    elif model_nr==11:
        epochs = 40
        if method >= 1:
            if trainings_different_model>1:
                epochs = 40
            else:
                epochs=40
    elif model_nr==12:
        epochs = 200
        if method >= 1:
            if trainings_different_model>1:
                epochs = 150
            else:
                epochs=150
    elif dataset=='imagenet_ood':
        epochs=10#50
        #print(epochs)
        if model_nr == 17 or model_nr==18:
            epochs = 70

    if not mini_batch:
        epochs=400

    #epochs=1

    model_save=False
    model_load=False

    iter_count=-1#-1
    with_optimizer_parameter_copy=None
    if method==1 or method==5 or method==6 or method==7 or method==8 or method==9 or method==11 or method==12 or method==21 or method==22 or method==23 or (method>=24 and method<=40):
        iter_count = 2#5#may be changed (>=2)
        # if iter_count>2 and trainings_same_model!=1:
        #     trainings_same_model=int(trainings_same_model/2)
    if method==2:
        with_optimizer_parameter_copy=False#default: False #it is not known which is better
        iter_count=1#don't change

    folds_num=None
    if cross_val:
        folds_num=trainings_same_model
        trainings_same_model=1

    # calculate_dist=True
    step_is_fraction_of_optimizer_denominator=0.#np.inf#np.inf#np.Inf#0.01 #0 value turns it off
    denominator_mul=None
    if step_is_fraction_of_optimizer_denominator!=0.:
        denominator_mul=.99

    d_type=torch.float#torch.double, or torch.float, etc.

    log_interval=10
    if dataset=='imagenet_ood':
        log_interval=1

    def pretraining_finish_criterion(stats,epoch):
        return stats and 'train_accuracy' in stats and stats['train_accuracy'][epoch][-1]>=55.

    pretraining=None#{'method':0,'method_after_pretraining':method,'pretraining_finish_criterion':pretraining_finish_criterion}#None

    if pretraining:
        pretraining['finished']=False

    #model_layer_graph=True


    def init_weights(layers,init_method=torch.nn.init.xavier_uniform,avg_gain=1.):
        for ind in range(0,len(layers)):
            if type(layers[ind])!=nn.BatchNorm2d:
                if hasattr(layers[ind],'weight'):
                    init_method(layers[ind].weight)
                    if avg_gain!=1.:
                        layers[ind].weight.requires_grad = False
                        layers[ind].weight*=avg_gain
                        layers[ind].weight.requires_grad = True

                if hasattr(layers[ind],'bias'):
                    #init_method(layers[ind].bias)
                    layers[ind].bias.requires_grad=False
                    layers[ind].bias.zero_()
                    layers[ind].bias.requires_grad=True

    def train(model, device, train_loader, optimizer, epoch,model2=None,opt2=None,log_interval=4,epoch_num=-1):
        model.train()

        #optimizer.param_groups[0]['lr']/=len(train_loader.dataset)
        if model2 is None or opt2 is None:
            model2 = NN(copy.deepcopy(model.layers))
            model2.train()
            model2.load_state_dict(model.state_dict().copy())
            opt2 = type(optimizer)(model2.parameters(),**optim_args)
            opt2.load_state_dict(optimizer.state_dict().copy())

        # model3 = NN(copy.deepcopy(model.layers))
        # model3.train()
        # model3.load_state_dict(model.state_dict().copy())
        # opt3 = type(optimizer)(model.parameters())
        # opt3.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']

        opt2.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']

        global iter_count

        model3=None
        if (method==1 or (method>=3)) and iter_count>=3:
            model3 = NN(copy.deepcopy(model.layers))
            model3.train()
            model3.load_state_dict(model.state_dict().copy())
            # opt3 = type(optimizer)(model.parameters())
            # opt3.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']
        #model.double()
        #model2.double()

        if method==0:
            optimizer.zero_grad()
            # if show_grad and model.layers[0].weight.grad is not None:
            #     print(model.layers[0].weight.grad[0][0][0])
            loss_sum_log=0.
            loss_count=0
            total_loss_sum=0.
            total_loss_count=0
            #torch.manual_seed(3)
            for batch_idx, (data, target) in enumerate(train_loader):
                #data, target = data.double().to(device), target.to(device)
                data, target = data.to(device), target.to(device)


                # data.requires_grad=True
                output = model(data)
                loss = F.cross_entropy(output, target,reduction='sum')  # F.nll_loss(output, target)
                # loss.requires_grad=True
                # print(hash(model.named_parameters()))
                # print(dict(model.named_parameters()).keys())

                # torch.manual_seed(1)
                loss.backward()
                # model.backward(loss)
                # torch.manual_seed(1)
                # x=model.state_dict()
                # print(x)
                loss_sum_log+=loss.item()
                loss_count+=data.size()[0]
                total_loss_sum+=loss.item()
                total_loss_count+=data.size()[0]
                if (batch_idx+1) % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * batch_size, len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss_sum_log/loss_count))
                    loss_sum_log=0.
                    loss_count=0

            optimizer.step()
            if show_grad:
                print(model.layers[0].weight.grad[0][0][0])
            if not calculate_train_stats:#then save training stats
                if 'train_loss_onthefly' not in model.stats:
                    model.stats['train_loss_onthefly'] = {}
                if epoch not in model.stats['train_loss_onthefly']:
                    model.stats['train_loss_onthefly'][epoch] = []
                if total_loss_count!=0:
                    model.stats['train_loss_onthefly'][epoch].append(total_loss_sum/total_loss_count)
        elif method==1:
            if iter_count>=3:
                model2.load_state_dict(model.state_dict())
                model3.load_state_dict(model.state_dict())
                opt2.zero_grad()

                loss_sum_log = 0.
                loss_count = 0
                total_loss_sum = 0.
                total_loss_count = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    # data, target = data.double().to(device), target.to(device)
                    data, target = data.to(device), target.to(device)
                    output = model2(data)
                    _loss = F.cross_entropy(output, target, reduction='sum')
                    _loss.backward()

                    loss_sum_log += _loss.item()
                    loss_count += data.size()[0]
                    total_loss_sum += _loss.item()
                    total_loss_count += data.size()[0]
                    if (batch_idx + 1) % log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * batch_size, len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader), loss_sum_log / loss_count))
                        loss_sum_log = 0.
                        loss_count = 0
                if show_grad:
                    print(model2.layers[0].weight.grad[0][0][0])


                # output = model2(data)
                # _loss = F.cross_entropy(output, target)  # F.nll_loss(output, target)

                # opt2.zero_grad()
                # _loss.backward()
                # if i == 0:
                #     opt2.zero_grad()
                #     _loss.backward()
                #     # model2.backward(_loss)
                #     # loss.backward(retain_graph=True)
                # else:
                #     model2.copy_grad_from(model)

                opt2.step()
                # _output = model2(data)
                # loss_backprop = F.cross_entropy(_output, target)

                for i in range(iter_count - 1):
                    # if i % 2 == 0:
                    #     if i != 0:
                    #         model.load_state_dict(model3.state_dict())
                    #         _output = model2(data)
                    #         loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)
                    #     else:
                    #         loss = loss_backprop
                    #
                    #     output1 = model(data)
                    #     loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)
                    #
                    #     model.backward_grad_correction_with_weight_change2(loss1, model2)
                    # else:
                        model2.load_state_dict(model.state_dict())
                        model.load_state_dict(model3.state_dict())

                        loss_sum_log = 0.
                        loss_count = 0
                        total_loss_sum = 0.
                        total_loss_count = 0
                        for batch_idx, (data, target) in enumerate(train_loader):
                            # data, target = data.double().to(device), target.to(device)
                            data, target = data.to(device), target.to(device)
                            # output = model2(data)
                            # _loss = F.cross_entropy(output, target, reduction='sum')
                            # _loss.backward()

                            _output = model(data)
                            loss = F.cross_entropy(_output, target, reduction='sum')  # F.nll_loss(output, target)

                            output1 = model2(data)
                            #loss1 = F.cross_entropy(output1, target, reduction='sum')  # F.nll_loss(output1, target)
                            weight_change = len(train_loader.dataset) == total_loss_count
                            if weight_change:
                                print('weight change')
                            model.backward_grad_correction_with_weight_change2(loss, model2, weight_change=weight_change,
                                                                               accumulate_gradients=True)

                            loss_sum_log += loss.item()
                            loss_count += data.size()[0]
                            total_loss_sum += loss.item()
                            total_loss_count += data.size()[0]
                            if (batch_idx + 1) % log_interval == 0:
                                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                    epoch, batch_idx * batch_size, len(train_loader.dataset),
                                           100. * batch_idx / len(train_loader), loss_sum_log / loss_count))
                                loss_sum_log = 0.
                                loss_count = 0
                        if show_grad:
                            print(model.layers[0].weight.grad[0][0][0])

                # if (iter_count) % 2 == 1:
                #     model.load_state_dict(model2.state_dict())

                # output2 = model(data)
                # loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)
                #
                # batch_counter += 1
                # if loss_backprop < loss2:
                #     higher_loss_batch_counter += 1
                #     # high_loss=True
                # elif loss_backprop > loss2:
                #     lower_loss_batch_counter += 1
                # loss_improvement += float(loss_backprop - loss2)
                #
                # if float(loss_backprop - _loss) != 0.:
                #     relative_loss_improvement += float((loss_backprop - loss2) / abs(loss_backprop - _loss))
            else:
                model2.load_state_dict(model.state_dict())
                opt2.zero_grad()

                loss_sum_log = 0.
                loss_count = 0
                total_loss_sum = 0.
                total_loss_count = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    # data, target = data.double().to(device), target.to(device)
                    data, target = data.to(device), target.to(device)
                    output = model2(data)
                    _loss = F.cross_entropy(output, target,reduction='sum')
                    _loss.backward()

                    loss_sum_log += _loss.item()
                    loss_count += data.size()[0]
                    total_loss_sum += _loss.item()
                    total_loss_count += data.size()[0]
                    if (batch_idx + 1) % log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * batch_size, len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader), loss_sum_log / loss_count))
                        loss_sum_log = 0.
                        loss_count = 0
                if show_grad:
                    print(model2.layers[0].weight.grad[0][0][0])
                if not calculate_train_stats:  # then save training stats
                    if 'train_loss_onthefly' not in model.stats:
                        model.stats['train_loss_onthefly'] = {}
                    if epoch not in model.stats['train_loss_onthefly']:
                        model.stats['train_loss_onthefly'][epoch] = []
                    if total_loss_count != 0:
                        model.stats['train_loss_onthefly'][epoch].append(total_loss_sum / total_loss_count)
                opt2.step()

                model.copy_grad_from(model2)#just to initialize tensors to hold gradients
                optimizer.zero_grad()#initialize gradients to 0.
                loss_sum_log = 0.
                loss_count = 0
                #total_loss_sum = 0.
                total_loss_count = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    # data, target = data.double().to(device), target.to(device)
                    data, target = data.to(device), target.to(device)

                    _output = model2(data)
                    loss = F.cross_entropy(_output, target,reduction='sum')#this line is just to log a loss

                    output1 = model(data)
                    loss1 = F.cross_entropy(output1, target,reduction='sum')

                    loss_sum_log += loss.item()
                    loss_count += data.size()[0]
                    #total_loss_sum += loss.item()
                    total_loss_count += data.size()[0]

                    weight_change=len(train_loader.dataset)==total_loss_count
                    if weight_change:
                        print('weight change')
                    model.backward_grad_correction_with_weight_change2(loss1, model2,weight_change=weight_change,accumulate_gradients=True)
                    # output2 = model(data)
                    # loss2 = F.cross_entropy(output2, target)

                    if (batch_idx + 1) % log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * batch_size, len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader), loss_sum_log / loss_count))
                        loss_sum_log = 0.
                        loss_count = 0
                if show_grad:
                    print(model.layers[0].weight.grad[0][0][0])



    total_relative_loss_improvement_denominator = 0.
    total_loss_improvement = 0.

    initialized=None
    model3=None
    opt3=None

    print_cuda_memory_for_some_methods = False #for method 22 with more than 2 iterations and method 23

    method33_index = 0
    method33_copy_freq = 40
    model_state_buffer = None

    def train_minibatch(model, device, train_loader, optimizer, epoch,model2=None,opt2=None,log_interval=10):
        global method33_index
        global model_state_buffer
        #torch.cuda.empty_cache()
        if method>20:
            assert model_layer_graph
        model.train()

        if model2 is None or opt2 is None:
            #model2 = NN_Advanced(copy.deepcopy(model.layers)) if method==4 else NN(copy.deepcopy(model.layers))
            #model2=type(model)(copy.deepcopy(model.layers))
            #model2.train()
            #model2.load_state_dict(model.state_dict().copy())
            model2=copy.deepcopy(model)
            opt2 = type(optimizer)(model2.parameters(),**optim_args)
            #opt2.load_state_dict(optimizer.state_dict().copy())
            opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))

        #model3 = NN_Advanced(copy.deepcopy(model.layers)) if method==4 else NN(copy.deepcopy(model.layers))#NN(copy.deepcopy(model.layers))
        #model3.train()
        #model3.load_state_dict(copy.deepcopy(model.state_dict()))
        #model3=None
        #opt3=None
        global iter_count
        global model3
        global opt3
        if iter_count > 2 or (method == 22 and hasattr(model, 'gradient_factor2')) or method==23 or method==30 or method==31 or method==32 or (method==37 and epoch==1):# or method==33:
            if model3 is None:
                model3 = copy.deepcopy(model)
            else:
                model3.load_state_dict(model.state_dict())
            if method < 20:
                if opt3 is None:
                    opt3 = type(optimizer)(model.parameters(), **optim_args)

                opt3.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']

        # model2.step_factor=model.step_factor
        # model2.gradient_factor=model.gradient_factor
        #opt2.lr=optimizer.lr

        global total_relative_loss_improvement_denominator
        global total_loss_improvement
        if epoch==1:
            total_relative_loss_improvement_denominator = 0.
            total_loss_improvement = 0.
        relative_loss_improvement_denominator=0.
        loss_improvement=0.
        higher_loss_batch_counter=0
        lower_loss_batch_counter=0
        batch_counter=0

        opt2.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']

        method_7_lr_mul=1
        if method==7:
            opt2.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*method_7_lr_mul

        common_optimizer_parameters_enhancement=False
        if common_optimizer_parameters_enhancement and method==1:
            opt2.param_groups[0]['alpha'] = optimizer.param_groups[0]['alpha']**0.5

        take_optimal_update=False

        lr_initial=None
        lr_including_momentum=None
        momentum_coef=None
        #betas=None#to store momentum, which is the first element of the 'betas' tuple
        momentum_key=None
        zero_momentum=None
        global initialized
        if method==10:
            if initialized is None:
                initialized = False
            #optimizer.param_groups[0]['momentum']=0.
            #opt2.param_groups[0]['momentum']=0.
            #momentum_coef=0.9
            # lr_initial = optimizer.param_groups[0]['lr']
            # lr_including_momentum=optimizer.param_groups[0]['lr']/(1-momentum_coef)
        elif method==11 or method==12:
            if initialized is None:
                initialized = False
            # optimizer.param_groups[0]['momentum']=0.
            # opt2.param_groups[0]['momentum']=0.
            #momentum_coef = 0.9
            momentum_coef=optim_args['momentum'] if method==11 else optim_args['betas'][0]
            lr_initial = optimizer.param_groups[0]['lr']
            lr_including_momentum = optimizer.param_groups[0]['lr'] / (1 - momentum_coef)
            if iter_count<=2:
                opt2.param_groups[0]['lr']=lr_including_momentum

            if method==11:
                momentum_key='momentum'
                zero_momentum=0.
            elif method==12:
                momentum_key='betas'
                #momentum_coef=optim_args['betas']
                zero_momentum=(0.,optim_args['betas'][1])


        mapping=None#class mapping specifically for mode1 nr 13
        logit_mask=None#logit mask specifically for model nr 13
        if model_nr==13 or model_nr==14 or model_nr==15 or model_nr==16 or model_nr==17 or model_nr==18:
            mapping=get_approx_optimal_class_mapping(train_loader)
            mapping=mapping.long().to(device)
            logit_mask=get_logit_mask(mapping)
            logit_mask = logit_mask.to(device)
        # if method==8:
        #     lr_initial=optimizer.param_groups[0]['lr']
        #     betas=optimizer.param_groups[0]['betas']
        #     lr_including_momentum=lr_initial/(1.-betas[0])
        #     # optimizer.param_groups[0]['lr']=lr_initial/(1.-optimizer.param_groups[0]['momentum'])
        #     # opt2.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']
        #torch.manual_seed(1)

        step_length_stats=method==27 or (True and (method==26 or method==22))
        print_step_length=True
        step_length_base_l1=0.
        step_length_base_l2=0.
        _step_length_base_l1=0.
        _step_length_base_l2=0.
        step_length_l1=0.
        step_length_l2=0.

        # method33_index=0
        # method33_copy_freq=40
        if method==33 or method==34:
            model2.load_state_dict(model.state_dict())
            #model3.load_state_dict(model)

        if method==37 or method==38:
            if epoch==1:
                model2.load_state_dict(model.state_dict())

                if not hasattr(model, 'copy_freq'):
                    setattr(model, 'copy_freq', 40)
                if method==38:
                    # if not hasattr(model,'copy_freq'):
                    #     setattr(model,'copy_freq',400)
                    if not hasattr(model,'state_buffer_size'):
                        setattr(model,'state_buffer_size',model.copy_freq/5)
                    model_state_buffer=ModelStateBuffer(model.state_buffer_size)


        if method == 35 or method==36:
            # # momentum_key = 'betas'
            # # # momentum_coef=optim_args['betas']
            # # zero_momentum = (0., optim_args['betas'][1])
            # if epoch==1:
            #     if 'betas' in opt2.param_groups[0]:
            #         betas=opt2.param_groups[0]['betas']
            #         opt2.param_groups[0]['betas']=(0,betas[1])
            if epoch==1:
                if 'betas' in opt2.param_groups[0]:
                    betas=opt2.param_groups[0]['betas']
                    opt2.param_groups[0]['betas']=(0.,0.99)#(0,betas[1])

                    optimizer.param_groups[0]['betas']=(0.,0.98)#(0,betas[1])
            optimizer.param_groups[0]['betas'],opt2.param_groups[0]['betas']=opt2.param_groups[0]['betas'],optimizer.param_groups[0]['betas']

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            additional_upgrade=False
            loss = None

            if method==41:#if method == 21 or method == 22 or (method >= 24 and method <= 29):

                m22_l2_norm = method == 22
                grad_l2_norm = None

                # for batch_idx, (data, target) in enumerate(train_loader):
                #     data, target = data.to(device), target.to(device)
                if mapping is not None:
                    target = mapping[target]

                model2.load_state_dict(model.state_dict())

                if method != 28 and method != 29:
                    if method == 24:
                        # copy_optimizer_params(opt2, optimizer, model2, model)
                        try:
                            copy_optimizer_params(opt2, optimizer, model2, model)
                        except:
                            opt2.load_state_dict(optimizer.state_dict().copy())
                    else:
                        try:
                            copy_optimizer_params(optimizer, opt2, model, model2)
                        except:
                            # optimizer.load_state_dict(opt2.state_dict().copy())
                            optimizer.load_state_dict(copy.deepcopy(opt2.state_dict()))

                if method == 28 or method == 29:
                    betas = optimizer.param_groups[0]['betas']
                    betas_zero_momentum = (0., betas[1])
                    optimizer.param_groups[0]['betas'] = betas_zero_momentum

                    if method == 28:
                        opt2.param_groups[0]['lr'] = 0.5 * optimizer.param_groups[0][
                            'lr']  # 10.0*optimizer.param_groups[0]['lr']

                # if with_optimizer_parameter_copy:
                #     opt2.load_state_dict(optimizer.state_dict())
                # opt2.load_state_dict(optimizer)

                model2.set_require_grad(False)

                loss_gradient_arithmetic_avg=None
                if method==41:
                    output = model2(data)

                    loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)

                    opt2.zero_grad()
                    loss.backward(retain_graph=False)

                    if m22_l2_norm:
                        grad_l2_norm = model2.grad_l2_norm()
                    opt2.step()

                    first_step_length_l1=model.step_length(model2,L=1)

                    output = model2(data)

                    loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)

                    # opt2.zero_grad()
                    loss.backward(retain_graph=False)

                    model.copy_grad_from(model2)
                    model2.load_state_dict(model.state_dict())
                    model2.copy_grad_from(model)
                    model.zero_grad()

                    # if m22_l2_norm:
                    #     grad_l2_norm = model2.grad_l2_norm()

                    try:
                        copy_optimizer_params(opt2, optimizer, model2, model)
                    except:
                        # optimizer.load_state_dict(opt2.state_dict().copy())
                        opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))

                    opt2.step()
                    second_step_length_l1 = model.step_length(model2, L=1)
                    if first_step_length_l1!=0 and second_step_length_l1!=0:
                        model2.mul_update(model,first_step_length_l1/second_step_length_l1)

                    output = model2(data)
                    loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)

                    loss_gradient_arithmetic_avg=loss.item()
                    try:
                        copy_optimizer_params(opt2, optimizer, model2, model)
                    except:
                        # optimizer.load_state_dict(opt2.state_dict().copy())
                        opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                    model2.load_state_dict(model.state_dict())

                # torch.cuda.empty_cache()
                # print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)
                # print('6. ' + str(torch.cuda.memory_allocated()))
                # if logit_mask is not None:
                #     output = output + logit_mask
                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                # with torch.no_grad():
                #     correct += torch.count_nonzero(target == torch.argmax(output, 1), 0)
                #     predicted += target.shape[0]

                # model2.set_require_grad(main_skip_connection_support.average_gradient_of_nonlinear_layers_enhancement)

                opt2.zero_grad()
                loss.backward(retain_graph=False)

                if m22_l2_norm:
                    grad_l2_norm = model2.grad_l2_norm()
                # if i == 0:
                #     opt2.zero_grad()
                #     loss.backward(retain_graph=False)
                #
                # else:
                #     model2.copy_grad_from(model)

                # print('7. ' + str(torch.cuda.memory_allocated()))
                opt2.step()

                del loss
                del output

                # if main_skip_connection_support.average_gradient_of_nonlinear_layers_enhancement:
                #     model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()

                output1 = model(data)
                # output1+=logit_mask
                # if logit_mask is not None:
                #     output1+=logit_mask
                # loss1 = torch.nn.functional.cross_entropy(output1,
                #                                           target)  # F.nll_loss(output1, target)
                loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                # loss1.requires_grad=True

                model2.set_require_grad(True)
                output = model2(data)
                # output += logit_mask
                # if logit_mask is not None:
                #     output+=logit_mask
                # loss = torch.nn.functional.cross_entropy(output,
                #                                          target)  # F.nll_loss(output, target)#todo check what this line changes
                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True

                if average_gradient_of_nonlinear_layers_enhancement and not average_gradient_of_linear_layers_enhancement:
                    model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()
                # if target.is_cuda:
                #     print(torch.cuda.memory_allocated())
                #     torch.cuda.empty_cache()
                #     print(torch.cuda.memory_allocated())
                # else:
                #
                #     process = psutil.Process()
                #     print(process.memory_info().rss)  # in bytes
                #     gc.collect()
                #     print(process.memory_info().rss)  # in bytes

                # print('1. ' + str(torch.cuda.memory_allocated()))
                if (method >= 22 and method <= 24):
                    if step_length_stats:
                        _step_length_base_l1 = model.step_length(model2, L=1)
                        _step_length_base_l2 = model.step_length(model2, L=2)
                        step_length_base_l1 += _step_length_base_l1
                        step_length_base_l2 += _step_length_base_l2

                    model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                       accumulate_gradients=False,
                                                                       gradient_only_modification=False)
                    # todo delete all not necessary grad tensors in backward_grad_correction_with_weight_change2

                    # print('2. ' + str(torch.cuda.memory_allocated()))
                    # model.avg_grad_to_integrated_grad(model2,model)
                    if m22_l2_norm:
                        avg_grad_l2_norm = model.grad_l2_norm()
                        if avg_grad_l2_norm != 0.:
                            model.mul_grad(grad_l2_norm / avg_grad_l2_norm)

                    if step_length_stats:
                        model2.load_state_dict(model.state_dict())

                    optimizer.step()
                    if method == 24:
                        model.load_state_dict(model2.state_dict())

                    if step_length_stats:
                        step_length_l1 += model.step_length(model2, L=1)
                        step_length_l2 += model.step_length(model2, L=2)

                        if print_step_length:
                            print('Method 22 step length L2: ' + str(
                                100. * step_length_l2 / step_length_base_l2 if step_length_base_l2 != 0. else 'Inf') + '%  L1: ' + str(
                                100. * step_length_l1 / step_length_base_l1 if step_length_base_l1 != 0. else 'Inf') + '%')

                elif method == 21 or method == 25:
                    if method == 21:
                        model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=True,
                                                                           accumulate_gradients=False,
                                                                           gradient_only_modification=False)
                    else:
                        model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                           accumulate_gradients=False,
                                                                           gradient_only_modification=False)
                        model.weight_change_as_model_but_directed_by_grad(model2)
                elif method == 28 or method == 29:
                    model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                       accumulate_gradients=False,
                                                                       gradient_only_modification=False)
                    optimizer.step()
                else:
                    # method 26 or 27
                    if step_length_stats:
                        # step_length_base_l1 += model.step_length(model2, L=1)
                        # step_length_base_l2 += model.step_length(model2, L=2)
                        _step_length_base_l1 = model.step_length(model2, L=1)
                        _step_length_base_l2 = model.step_length(model2, L=2)
                        step_length_base_l1 += _step_length_base_l1
                        step_length_base_l2 += _step_length_base_l2

                    model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                       accumulate_gradients=False,
                                                                       gradient_only_modification=False)
                    model.layerwise_grad_normalization(model2, L=2)
                    if step_length_stats:
                        model2.load_state_dict(model.state_dict())

                    optimizer.step()

                    if step_length_stats:
                        if method == 27 or method==41:
                            _step_length_l1 = model.step_length(model2, L=1)
                            _step_length_l2 = model.step_length(model2, L=2)
                            if _step_length_l1 != 0:
                                model.mul_update(model2, _step_length_base_l1 / _step_length_l1)
                            # if _step_length_l2!=0 and _step_length_l1!=0:
                            #     model.mul_update(model2,_step_length_base_l2/_step_length_l2)
                            # step_length_l1 += _step_length_l1
                            # step_length_l2 += _step_length_l2
                            step_length_l1 += model.step_length(model2, L=1)
                            step_length_l2 += model.step_length(model2, L=2)
                        else:
                            step_length_l1 = model.step_length(model2, L=1)
                            step_length_l2 = model.step_length(model2, L=2)

                        if print_step_length:
                            print('Method 26 step length L2: ' + str(
                                100. * step_length_l2 / step_length_base_l2 if step_length_base_l2 != 0. else 'Inf') + '%  L1: ' + str(
                                100. * step_length_l1 / step_length_base_l1 if step_length_base_l1 != 0. else 'Inf') + '%')
                # print('3. ' + str(torch.cuda.memory_allocated()))
                # !!!!Largest cuda memory consumption
                # with open("MemoryLog.txt", "a") as myfile:
                #     myfile.write(str(torch.cuda.memory_allocated())+',\n')

                loss_to_del = loss
                loss1_to_del = loss1
                loss = loss.item()
                loss1 = loss1.item()
                del loss_to_del, loss1_to_del
                del output, output1
                model.input_output_cleanup()
                # print('3.2. ' + str(torch.cuda.memory_allocated()))

                loss2 = None
                with torch.no_grad():
                    output2 = model(data)
                    # if logit_mask is not None:
                    #     output2 += logit_mask
                    # loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)
                    loss2 = F.cross_entropy(output2 if logit_mask is None else output2 + logit_mask, target)
                    # print('4. ' + str(torch.cuda.memory_allocated()))

                batch_counter += 1
                #loss=loss_gradient_arithmetic_avg
                if loss < loss2:
                    higher_loss_batch_counter += 1
                    high_loss = True
                    # model.load_state_dict(model2.state_dict())
                elif loss > loss2:
                    lower_loss_batch_counter += 1
                loss_improvement += float(loss - loss2)

                # if float(loss - loss1) != 0.:
                #     relative_loss_improvement += float((loss - loss2) / abs(loss - loss1))
                relative_loss_improvement_denominator += float(loss - loss1)

                total_loss_improvement += float(loss - loss2)
                total_relative_loss_improvement_denominator += float(loss - loss1)

                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss_initial))
                    if batch_counter != 0:
                        print("Higher loss batches: " + str(higher_loss_batch_counter) + "/" + str(
                            batch_counter) + "=" + str(
                            higher_loss_batch_counter / batch_counter) + " Lower loss batch ratio: " + str(
                            lower_loss_batch_counter / batch_counter) + " Avg loss improvement: " + str(
                            loss_improvement / batch_counter) + " Avg relative loss improvement: " + str(
                            (loss_improvement / abs(
                                relative_loss_improvement_denominator)) if relative_loss_improvement_denominator != 0 else "Division by zero") + " Total avg relative loss improvement: " + str(
                            total_loss_improvement / abs(
                                total_relative_loss_improvement_denominator) if total_relative_loss_improvement_denominator != 0 else "Division by zero"))

                del output2, loss2
                model.input_output_cleanup()
                # print('4.2. ' + str(torch.cuda.memory_allocated()))
                continue
            elif method == 38:
                global method33_index
                if mapping is not None:
                    target = mapping[target]

                # model2.load_state_dict(model3.state_dict())
                # model3.load_state_dict(model.state_dict())

                model2.set_require_grad(True)
                # torch.cuda.empty_cache()
                # print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)

                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                opt2.zero_grad()
                loss.backward(retain_graph=True)

                del loss
                del output

                output1 = model(data)

                loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                loss = loss1.item()

                if average_gradient_of_nonlinear_layers_enhancement and not average_gradient_of_linear_layers_enhancement:
                    model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()

                model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                   accumulate_gradients=False,
                                                                   gradient_only_modification=False)
                optimizer.step()

                loss1_to_del = loss1
                # loss = loss.item()
                loss1 = loss1.item()
                # del loss_to_del, loss1_to_del
                del loss1_to_del
                # del output, output1
                del output1
                model.input_output_cleanup()

                if method33_index % int(model.copy_freq/model.state_buffer_size) == 0:
                    # model2.load_state_dict(model.state_dict())
                    if len(model_state_buffer.content)!=0:
                        model2.load_state_dict(model_state_buffer.get_item())
                        model_state_buffer.add(model.state_dict())
                    else:
                        model_state_buffer.add(model.state_dict())
                        model2.load_state_dict(model.state_dict())


                method33_index += 1
            elif method == 37:
                # global method33_index
                if mapping is not None:
                    target = mapping[target]

                # model2.load_state_dict(model3.state_dict())
                # model3.load_state_dict(model.state_dict())

                model2.set_require_grad(True)
                # torch.cuda.empty_cache()
                # print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)

                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                opt2.zero_grad()
                loss.backward(retain_graph=True)

                del loss
                del output

                output1 = model(data)

                loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                loss = loss1.item()

                if average_gradient_of_nonlinear_layers_enhancement and not average_gradient_of_linear_layers_enhancement:
                    model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()

                # #just for statistics:
                # stats=False
                # if stats:
                #     model3.load_state_dict(model.state_dict())

                model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                   accumulate_gradients=False,
                                                                   gradient_only_modification=False)
                optimizer.step()

                # if stats:
                #     dist=model.distance_from(model3,l=1)
                #     print(str(method33_index)+": "+str(dist))


                loss1_to_del = loss1
                # loss = loss.item()
                loss1 = loss1.item()
                # del loss_to_del, loss1_to_del
                del loss1_to_del
                # del output, output1
                del output1
                model.input_output_cleanup()

                # if method33_index % (method33_copy_freq/2) == 0:
                if method33_index % (model.copy_freq / 2) == 0:
                    model2.load_state_dict(model3.state_dict())
                    model3.load_state_dict(model.state_dict())
                # if method33_index % method33_copy_freq == 0:
                #     model2.load_state_dict(model.state_dict())
                method33_index += 1
            elif method==36:
                if mapping is not None:
                    target = mapping[target]

                # model2.load_state_dict(model3.state_dict())
                # model3.load_state_dict(model.state_dict())


                #model2.set_require_grad(True)
                # torch.cuda.empty_cache()
                # print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)
                # print('6. ' + str(torch.cuda.memory_allocated()))
                # if logit_mask is not None:
                #     output = output + logit_mask
                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                opt2.zero_grad()
                loss.backward()
                #opt2.step()

                #############################################################################################
                # del loss
                # del output


                output1 = model(data)

                loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                __loss=loss1.item()
                # optimizer.zero_grad()
                # model.backward()
                model.copy_grad_from(model2)
                loss1.backward()
                #optimizer.step()
                model2.copy_grad_from(model)
                optimizer.step()
                opt2.step()


                # model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                #                                                    accumulate_gradients=False,
                #                                                    gradient_only_modification=False)
                # optimizer.step()

                # optimizer.zero_grad()
                # model.set_require_grad(True)
                # output1 = model(data)
                # loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                # loss1.backward(retain_graph=True)
                # model2.backward_grad_correction_with_weight_change2(loss, model, weight_change=False,
                #                                                    accumulate_gradients=False,
                #                                                    gradient_only_modification=False)
                # opt2.step()

                with torch.no_grad():
                    _output=model2(data)
                    _loss=F.cross_entropy(_output if logit_mask is None else _output + logit_mask, target)

                    _output1 = model(data)
                    _loss1 = F.cross_entropy(_output1 if logit_mask is None else _output1 + logit_mask, target)

                    dist=model.distance_from(model2,l=1)
                    print(str(dist))
                    max_dist=5.
                    if dist>max_dist:
                        # model.mul_update(model2,1.-(dist-max_dist)/dist/2.)
                        # dist=(dist-max_dist)/2.+max_dist
                        # model2.mul_update(model,1.-(dist-max_dist)/dist)
                        if _loss<_loss1:
                            model.mul_update(model2, 1. - (dist - max_dist) / dist)
                        else:
                            model2.mul_update(model,1.-(dist-max_dist)/dist)

                loss1_to_del = loss1
                # loss = loss.item()
                loss1 = loss1.item()
                # del loss_to_del, loss1_to_del
                del loss1_to_del
                # del output, output1
                del output1
                model.input_output_cleanup()
                loss=__loss
                #model,model2=model2,model
            elif method==35:
                if mapping is not None:
                    target = mapping[target]

                # model2.load_state_dict(model3.state_dict())
                # model3.load_state_dict(model.state_dict())


                model2.set_require_grad(True)
                # torch.cuda.empty_cache()
                # print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)
                # print('6. ' + str(torch.cuda.memory_allocated()))
                # if logit_mask is not None:
                #     output = output + logit_mask
                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                opt2.zero_grad()
                loss.backward(retain_graph=True)

                #############################################################################################
                # del loss
                # del output


                output1 = model(data)

                loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                __loss=loss1.item()


                # if average_gradient_of_nonlinear_layers_enhancement and not average_gradient_of_linear_layers_enhancement:
                #     model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()


                model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                   accumulate_gradients=False,
                                                                   gradient_only_modification=False)
                optimizer.step()

                optimizer.zero_grad()
                model.set_require_grad(True)
                output1 = model(data)
                loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                loss1.backward(retain_graph=True)
                model2.backward_grad_correction_with_weight_change2(loss, model, weight_change=False,
                                                                   accumulate_gradients=False,
                                                                   gradient_only_modification=False)
                opt2.step()

                with torch.no_grad():
                    _output=model2(data)
                    _loss=F.cross_entropy(_output if logit_mask is None else _output + logit_mask, target)


                    dist=model.distance_from(model2,l=1)
                    print(str(dist))
                    max_dist=5.
                    if dist>max_dist:
                        # model.mul_update(model2,1.-(dist-max_dist)/dist/2.)
                        # dist=(dist-max_dist)/2.+max_dist
                        # model2.mul_update(model,1.-(dist-max_dist)/dist)
                        if _loss<loss1:
                            model.mul_update(model2, 1. - (dist - max_dist) / dist)
                        else:
                            model2.mul_update(model,1.-(dist-max_dist)/dist)

                loss1_to_del = loss1
                # loss = loss.item()
                loss1 = loss1.item()
                # del loss_to_del, loss1_to_del
                del loss1_to_del
                # del output, output1
                del output1
                model.input_output_cleanup()
                loss=__loss
                #model,model2=model2,model
            elif method==34:
                if mapping is not None:
                    target = mapping[target]

                # model2.load_state_dict(model3.state_dict())
                # model3.load_state_dict(model.state_dict())


                model2.set_require_grad(True)
                # torch.cuda.empty_cache()
                # print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)

                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                opt2.zero_grad()
                loss.backward(retain_graph=True)


                del loss
                del output


                output1 = model(data)

                loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                loss = loss1.item()

                if average_gradient_of_nonlinear_layers_enhancement and not average_gradient_of_linear_layers_enhancement:
                    model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()


                model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                   accumulate_gradients=False,
                                                                   gradient_only_modification=True)
                optimizer.step()

                loss1_to_del = loss1
                # loss = loss.item()
                loss1 = loss1.item()
                # del loss_to_del, loss1_to_del
                del loss1_to_del
                # del output, output1
                del output1
                model.input_output_cleanup()

                if method33_index%method33_copy_freq==0:
                    model2.load_state_dict(model.state_dict())
                method33_index+=1
            elif method==33:
                if mapping is not None:
                    target = mapping[target]

                # model2.load_state_dict(model3.state_dict())
                # model3.load_state_dict(model.state_dict())


                model2.set_require_grad(True)
                # torch.cuda.empty_cache()
                # print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)

                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                opt2.zero_grad()
                loss.backward(retain_graph=True)


                del loss
                del output


                output1 = model(data)

                loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                loss = loss1.item()

                if average_gradient_of_nonlinear_layers_enhancement and not average_gradient_of_linear_layers_enhancement:
                    model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()


                model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                   accumulate_gradients=False,
                                                                   gradient_only_modification=False)
                optimizer.step()

                loss1_to_del = loss1
                # loss = loss.item()
                loss1 = loss1.item()
                # del loss_to_del, loss1_to_del
                del loss1_to_del
                # del output, output1
                del output1
                model.input_output_cleanup()

                if method33_index%method33_copy_freq==0:
                    model2.load_state_dict(model.state_dict())
                method33_index+=1
            elif method==32:
                # m22_l2_norm = method == 22
                grad_l2_norm = None

                # for batch_idx, (data, target) in enumerate(train_loader):
                #     data, target = data.to(device), target.to(device)
                if mapping is not None:
                    target = mapping[target]

                model2.load_state_dict(model3.state_dict())
                model3.load_state_dict(model.state_dict())


                model2.set_require_grad(False)
                # torch.cuda.empty_cache()
                # print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)
                # print('6. ' + str(torch.cuda.memory_allocated()))
                # if logit_mask is not None:
                #     output = output + logit_mask
                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                opt2.zero_grad()
                #model2.zero_grad()
                loss.backward(retain_graph=False)

                #opt2.step()

                del loss
                del output

                output1 = model(data)

                loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                loss = loss1.item()

                # if average_gradient_of_nonlinear_layers_enhancement and not average_gradient_of_linear_layers_enhancement:
                #     model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()
                #
                # model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                #                                                    accumulate_gradients=False,
                #                                                    gradient_only_modification=False)
                # model2.input_output_cleanup()
                optimizer.zero_grad()
                #####################################################################################################
                # model.copy_grad_from(model2)
                loss1.backward()
                optimizer.step()
                dist1=model.distance_from(model3)

                model.copy_grad_from(model2)
                optimizer.step()
                if dist1!=0.:
                    dist2=model.distance_from(model3)
                    if dist2!=0.:
                        model.mul_update(model3,2.*dist1/dist2)


                loss1_to_del = loss1
                # loss = loss.item()
                loss1 = loss1.item()
                # del loss_to_del, loss1_to_del
                del loss1_to_del
                # del output, output1
                del output1
                model.input_output_cleanup()
            elif method==31:
                # m22_l2_norm = method == 22
                grad_l2_norm = None

                # for batch_idx, (data, target) in enumerate(train_loader):
                #     data, target = data.to(device), target.to(device)
                if mapping is not None:
                    target = mapping[target]

                model2.load_state_dict(model3.state_dict())
                model3.load_state_dict(model.state_dict())


                model2.set_require_grad(False)
                # torch.cuda.empty_cache()
                # print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)
                # print('6. ' + str(torch.cuda.memory_allocated()))
                # if logit_mask is not None:
                #     output = output + logit_mask
                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                #opt2.zero_grad()
                model2.zero_grad()
                loss.backward(retain_graph=False)


                del loss
                del output

                output1 = model(data)

                loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                loss = loss1.item()

                # if average_gradient_of_nonlinear_layers_enhancement and not average_gradient_of_linear_layers_enhancement:
                #     model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()
                #
                # model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                #                                                    accumulate_gradients=False,
                #                                                    gradient_only_modification=False)
                # model2.input_output_cleanup()
                optimizer.zero_grad()
                #####################################################################################################
                model.copy_grad_from(model2)
                loss1.backward()
                optimizer.step()

                loss1_to_del = loss1
                # loss = loss.item()
                loss1 = loss1.item()
                # del loss_to_del, loss1_to_del
                del loss1_to_del
                # del output, output1
                del output1
                model.input_output_cleanup()
            elif method==30:#method == 21 or method == 22 or (method >= 24 and method <= 29):

                # m22_l2_norm = method == 22
                grad_l2_norm = None

                # for batch_idx, (data, target) in enumerate(train_loader):
                #     data, target = data.to(device), target.to(device)
                if mapping is not None:
                    target = mapping[target]

                model2.load_state_dict(model3.state_dict())
                model3.load_state_dict(model.state_dict())
                # model,model2=model2,model

                # model2.load_state_dict(model.state_dict())

                # if method != 28 and method != 29:
                #     if method == 24:
                #         # copy_optimizer_params(opt2, optimizer, model2, model)
                #         try:
                #             copy_optimizer_params(opt2, optimizer, model2, model)
                #         except:
                #             opt2.load_state_dict(optimizer.state_dict().copy())
                #     else:
                #         try:
                #             copy_optimizer_params(optimizer, opt2, model, model2)
                #         except:
                #             # optimizer.load_state_dict(opt2.state_dict().copy())
                #             optimizer.load_state_dict(copy.deepcopy(opt2.state_dict()))

                # if method == 28 or method == 29:
                #     betas = optimizer.param_groups[0]['betas']
                #     betas_zero_momentum = (0., betas[1])
                #     optimizer.param_groups[0]['betas'] = betas_zero_momentum
                #
                #     if method == 28:
                #         opt2.param_groups[0]['lr'] = 0.5 * optimizer.param_groups[0][
                #             'lr']  # 10.0*optimizer.param_groups[0]['lr']

                # if with_optimizer_parameter_copy:
                #     opt2.load_state_dict(optimizer.state_dict())
                # opt2.load_state_dict(optimizer)

                model2.set_require_grad(True)
                # torch.cuda.empty_cache()
                # print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)
                # print('6. ' + str(torch.cuda.memory_allocated()))
                # if logit_mask is not None:
                #     output = output + logit_mask
                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                # with torch.no_grad():
                #     correct += torch.count_nonzero(target == torch.argmax(output, 1), 0)
                #     predicted += target.shape[0]

                # model2.set_require_grad(main_skip_connection_support.average_gradient_of_nonlinear_layers_enhancement)

                opt2.zero_grad()
                loss.backward(retain_graph=True)

                # if m22_l2_norm:
                #     grad_l2_norm = model2.grad_l2_norm()
                # if i == 0:
                #     opt2.zero_grad()
                #     loss.backward(retain_graph=False)
                #
                # else:
                #     model2.copy_grad_from(model)

                # print('7. ' + str(torch.cuda.memory_allocated()))

                # opt2.step()

                del loss
                del output

                # if main_skip_connection_support.average_gradient_of_nonlinear_layers_enhancement:
                #     model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()

                output1 = model(data)
                # output1+=logit_mask
                # if logit_mask is not None:
                #     output1+=logit_mask
                # loss1 = torch.nn.functional.cross_entropy(output1,
                #                                           target)  # F.nll_loss(output1, target)
                loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                loss=loss1.item()
                # loss1.requires_grad=True

                # model2.set_require_grad(True)
                # output = model2(data)

                # output += logit_mask
                # if logit_mask is not None:
                #     output+=logit_mask
                # loss = torch.nn.functional.cross_entropy(output,
                #                                          target)  # F.nll_loss(output, target)#todo check what this line changes
                # loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True

                if average_gradient_of_nonlinear_layers_enhancement and not average_gradient_of_linear_layers_enhancement:
                    model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()
                # if target.is_cuda:
                #     print(torch.cuda.memory_allocated())
                #     torch.cuda.empty_cache()
                #     print(torch.cuda.memory_allocated())
                # else:
                #
                #     process = psutil.Process()
                #     print(process.memory_info().rss)  # in bytes
                #     gc.collect()
                #     print(process.memory_info().rss)  # in bytes

                model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                   accumulate_gradients=False,
                                                                   gradient_only_modification=False)
                optimizer.step()

                # print('1. ' + str(torch.cuda.memory_allocated()))
                # if (method >= 22 and method <= 24):
                #     if step_length_stats:
                #         _step_length_base_l1 = model.step_length(model2, L=1)
                #         _step_length_base_l2 = model.step_length(model2, L=2)
                #         step_length_base_l1 += _step_length_base_l1
                #         step_length_base_l2 += _step_length_base_l2
                #
                #     model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                #                                                        accumulate_gradients=False,
                #                                                        gradient_only_modification=False)
                #     # todo delete all not necessary grad tensors in backward_grad_correction_with_weight_change2
                #
                #     # print('2. ' + str(torch.cuda.memory_allocated()))
                #     # model.avg_grad_to_integrated_grad(model2,model)
                #     if m22_l2_norm:
                #         avg_grad_l2_norm = model.grad_l2_norm()
                #         if avg_grad_l2_norm != 0.:
                #             model.mul_grad(grad_l2_norm / avg_grad_l2_norm)
                #
                #     if step_length_stats:
                #         model2.load_state_dict(model.state_dict())
                #
                #     optimizer.step()
                #     if method == 24:
                #         model.load_state_dict(model2.state_dict())
                #
                #     if step_length_stats:
                #         step_length_l1 += model.step_length(model2, L=1)
                #         step_length_l2 += model.step_length(model2, L=2)
                #
                #         if print_step_length:
                #             print('Method 22 step length L2: ' + str(
                #                 100. * step_length_l2 / step_length_base_l2 if step_length_base_l2 != 0. else 'Inf') + '%  L1: ' + str(
                #                 100. * step_length_l1 / step_length_base_l1 if step_length_base_l1 != 0. else 'Inf') + '%')
                #
                # elif method == 21 or method == 25:
                #     if method == 21:
                #         model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=True,
                #                                                            accumulate_gradients=False,
                #                                                            gradient_only_modification=False)
                #     else:
                #         model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                #                                                            accumulate_gradients=False,
                #                                                            gradient_only_modification=False)
                #         model.weight_change_as_model_but_directed_by_grad(model2)
                # elif method == 28 or method == 29:
                #     model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                #                                                        accumulate_gradients=False,
                #                                                        gradient_only_modification=False)
                #     optimizer.step()
                # else:
                #     # method 26 or 27
                #     if step_length_stats:
                #         # step_length_base_l1 += model.step_length(model2, L=1)
                #         # step_length_base_l2 += model.step_length(model2, L=2)
                #         _step_length_base_l1 = model.step_length(model2, L=1)
                #         _step_length_base_l2 = model.step_length(model2, L=2)
                #         step_length_base_l1 += _step_length_base_l1
                #         step_length_base_l2 += _step_length_base_l2
                #
                #     model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                #                                                        accumulate_gradients=False,
                #                                                        gradient_only_modification=False)
                #     model.layerwise_grad_normalization(model2, L=2)
                #     if step_length_stats:
                #         model2.load_state_dict(model.state_dict())
                #
                #     optimizer.step()
                #
                #     if step_length_stats:
                #         if method == 27:
                #             _step_length_l1 = model.step_length(model2, L=1)
                #             _step_length_l2 = model.step_length(model2, L=2)
                #             if _step_length_l1 != 0:
                #                 model.mul_update(model2, _step_length_base_l1 / _step_length_l1)
                #             # if _step_length_l2!=0 and _step_length_l1!=0:
                #             #     model.mul_update(model2,_step_length_base_l2/_step_length_l2)
                #             # step_length_l1 += _step_length_l1
                #             # step_length_l2 += _step_length_l2
                #             step_length_l1 += model.step_length(model2, L=1)
                #             step_length_l2 += model.step_length(model2, L=2)
                #         else:
                #             step_length_l1 = model.step_length(model2, L=1)
                #             step_length_l2 = model.step_length(model2, L=2)
                #
                #         if print_step_length:
                #             print('Method 26 step length L2: ' + str(
                #                 100. * step_length_l2 / step_length_base_l2 if step_length_base_l2 != 0. else 'Inf') + '%  L1: ' + str(
                #                 100. * step_length_l1 / step_length_base_l1 if step_length_base_l1 != 0. else 'Inf') + '%')
                # # print('3. ' + str(torch.cuda.memory_allocated()))
                # # !!!!Largest cuda memory consumption
                # # with open("MemoryLog.txt", "a") as myfile:
                # #     myfile.write(str(torch.cuda.memory_allocated())+',\n')

                # loss_to_del = loss
                loss1_to_del = loss1
                # loss = loss.item()
                loss1 = loss1.item()
                # del loss_to_del, loss1_to_del
                del loss1_to_del
                # del output, output1
                del output1
                model.input_output_cleanup()
                # print('3.2. ' + str(torch.cuda.memory_allocated()))

                # loss2 = None
                # with torch.no_grad():
                #     output2 = model(data)
                #     # if logit_mask is not None:
                #     #     output2 += logit_mask
                #     # loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)
                #     loss2 = F.cross_entropy(output2 if logit_mask is None else output2 + logit_mask, target)
                #     # print('4. ' + str(torch.cuda.memory_allocated()))

                # batch_counter += 1
                # if loss < loss2:
                #     higher_loss_batch_counter += 1
                #     high_loss = True
                #     # model.load_state_dict(model2.state_dict())
                # elif loss > loss2:
                #     lower_loss_batch_counter += 1
                # loss_improvement += float(loss - loss2)
                #
                # # if float(loss - loss1) != 0.:
                # #     relative_loss_improvement += float((loss - loss2) / abs(loss - loss1))
                # relative_loss_improvement_denominator += float(loss - loss1)
                #
                # total_loss_improvement += float(loss - loss2)
                # total_relative_loss_improvement_denominator += float(loss - loss1)
                #
                # if batch_idx % log_interval == 0:
                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(data), len(train_loader.dataset),
                #                100. * batch_idx / len(train_loader), loss_initial))
                #     if batch_counter != 0:
                #         print("Higher loss batches: " + str(higher_loss_batch_counter) + "/" + str(
                #             batch_counter) + "=" + str(
                #             higher_loss_batch_counter / batch_counter) + " Lower loss batch ratio: " + str(
                #             lower_loss_batch_counter / batch_counter) + " Avg loss improvement: " + str(
                #             loss_improvement / batch_counter) + " Avg relative loss improvement: " + str(
                #             (loss_improvement / abs(
                #                 relative_loss_improvement_denominator)) if relative_loss_improvement_denominator != 0 else "Division by zero") + " Total avg relative loss improvement: " + str(
                #             total_loss_improvement / abs(
                #                 total_relative_loss_improvement_denominator) if total_relative_loss_improvement_denominator != 0 else "Division by zero"))

                # del output2, loss2
                # model.input_output_cleanup()
                # print('4.2. ' + str(torch.cuda.memory_allocated()))
                # continue
            elif method==23:# and iter_count>2:
                # 1.0074 = 40^(1/500)
                # 1.015 ~= 40^(1/250)
                #gradient_factor2_mul = 1.0074 ** (1 / ((iter_count - 2) if iter_count>2. else iter_count -1))
                gradient_factor2_mul = 1.015 ** (1 / ((iter_count - 2) if iter_count > 2. else iter_count - 1))
                gradient_factor2_decrease_power=6/3#9#means that 1 loss increase is accepted after average-gradient update per 9 loss decreases (lower_loss_batch_ratio~=0.9)
                if hasattr(model, 'gradient_factor2_decrease_power'):
                    gradient_factor2_decrease_power=model.gradient_factor2_decrease_power
                #gradient_factor2_mul = (40**(90/(gradient_factor2_decrease_power**0.85)/500)) ** (1 / ((iter_count - 2) if iter_count > 2. else iter_count - 1))
                epsilon=torch.finfo(torch.float32).eps
                max_gradient_factor2=1./(iter_count - 1)**(2./3.)
                if hasattr(model, 'max_gradient_factor2'):
                    max_gradient_factor2=model.max_gradient_factor2
                verbose=True
                verbose_str=''

                l1_norm_grad=True
                l1_norm_LR=False

                if mapping is not None:
                    target = mapping[target]

                if iter_count > 2:
                    model3.load_state_dict(model.state_dict())

                model2.load_state_dict(model.state_dict())

                # if with_optimizer_parameter_copy:
                #     opt2.load_state_dict(optimizer.state_dict())

                model2.set_require_grad(False)
                # torch.cuda.empty_cache()
                if print_cuda_memory_for_some_methods:
                    print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)
                # print('6. ' + str(torch.cuda.memory_allocated()))
                # if logit_mask is not None:
                #     output = output + logit_mask
                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                # with torch.no_grad():
                #     correct += torch.count_nonzero(target == torch.argmax(output, 1), 0)
                #     predicted += target.shape[0]

                # model2.set_require_grad(main_skip_connection_support.average_gradient_of_nonlinear_layers_enhancement)
                #opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))

                copy_optimizer_params(opt2,optimizer,model2,model)

                opt2.zero_grad()
                loss.backward(retain_graph=False)

                #########################################
                model3.copy_grad_from(model2)
                backprop_grad_l1_norm = model3.grad_l1_norm()
                # if i == 0:
                #     opt2.zero_grad()
                #     loss.backward(retain_graph=False)
                #
                # else:
                #     model2.copy_grad_from(model)

                # print('7. ' + str(torch.cuda.memory_allocated()))
                opt2.step()

                del loss
                del output

                loss_backprop = -torch.inf

                loss_ag = None


                prev_iter_loss=None

                #loss_for_display_stats=None
                # if main_skip_connection_support.average_gradient_of_nonlinear_layers_enhancement:
                #     model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()
                for iter in range(1, iter_count):
                    if iter != 1:
                        model2.load_state_dict(model.state_dict())
                        model.load_state_dict(model3.state_dict())

                    output1 = model(data)
                    # output1+=logit_mask
                    # if logit_mask is not None:
                    #     output1+=logit_mask
                    # loss1 = torch.nn.functional.cross_entropy(output1,
                    #                                           target)  # F.nll_loss(output1, target)
                    loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                    # loss1.requires_grad=True

                    model2.set_require_grad(True)
                    output = model2(data)
                    # output += logit_mask
                    # if logit_mask is not None:
                    #     output+=logit_mask
                    # loss = torch.nn.functional.cross_entropy(output,
                    #                                          target)  # F.nll_loss(output, target)#todo check what this line changes
                    loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                    # loss.requires_grad = True
                    loss_for_display_stats=loss.item()
                    if iter == 1:
                        loss_backprop = loss_for_display_stats
                        prev_iter_loss=loss_backprop
                    else:
                        #gradient_factor2_mul
                        if prev_iter_loss < loss_for_display_stats:#higher loss
                            if iter>1:#change gradientfactor2 only during further iterations, because the first may be inoptimal because of using different optimizer hyperparameters than the first update after the first backpropagation
                                model.gradient_factor2=model.gradient_factor2/(gradient_factor2_mul**gradient_factor2_decrease_power)
                            if verbose:
                                verbose_str+='H'
                        elif prev_iter_loss > loss_for_display_stats:#lower loss
                            if iter > 1:#change gradientfactor2 only during further iterations, because the first may be inoptimal because of using different optimizer hyperparameters than the first update after the first backpropagation
                                model.gradient_factor2=model.gradient_factor2*gradient_factor2_mul
                            if verbose:
                                verbose_str+='L'
                        prev_iter_loss=loss_for_display_stats
                        model.gradient_factor2 = min(max(model.gradient_factor2, epsilon), max_gradient_factor2)

                    if average_gradient_of_nonlinear_layers_enhancement and not average_gradient_of_linear_layers_enhancement:
                        model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()
                    # if target.is_cuda:
                    #     print(torch.cuda.memory_allocated())
                    #     torch.cuda.empty_cache()
                    #     print(torch.cuda.memory_allocated())
                    # else:
                    #
                    #     process = psutil.Process()
                    #     print(process.memory_info().rss)  # in bytes
                    #     gc.collect()
                    #     print(process.memory_info().rss)  # in bytes

                    # print('1. ' + str(torch.cuda.memory_allocated()))
                    #if method == 22:
                    if iter_count>2:
                        if type(optimizer)==optim.RMSprop:
                            optimizer.param_groups[0]['alpha']=opt2.param_groups[0]['alpha']**(1./(iter_count-1))
                        elif type(optimizer)==optim.Adam:
                            betas=opt2.param_groups[0]['betas']
                            optimizer.param_groups[0]['betas']=(betas[0],betas[1]**(1./(iter_count-1)))
                    #optimizer.load_state_dict(opt2.state_dict())
                    # optimizer.zero_grad()
                    model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                       accumulate_gradients=False,
                                                                       gradient_only_modification=False)
                    # todo delete all not necessary grad tensors in backward_grad_correction_with_weight_change2

                    grad_l1_norm=None
                    # print('2. ' + str(torch.cuda.memory_allocated()))
                    if hasattr(model,'gradient_factor2') and model.gradient_factor2!=1.:
                        model.soft_copy_grad(model3,1.-model.gradient_factor2)
                        if iter<iter_count-1:
                            model3.copy_grad_from(model)

                        #if iter_count>2:
                        if l1_norm_grad or l1_norm_LR:
                            grad_l1_norm = model.grad_l1_norm()
                            if l1_norm_grad:
                                #grad_l1_norm=model.grad_l1_norm()
                                if grad_l1_norm!=0.:
                                    model.mul_grad(backprop_grad_l1_norm/grad_l1_norm)
                            if l1_norm_LR:
                                optimizer.param_groups[0]['lr']=backprop_grad_l1_norm/grad_l1_norm*opt2.param_groups[0]['lr']
                    #model.soft_copy_grad(model3, 1. - iter / iter_count)
                    # model.soft_copy_grad(model3,1.-iter/(iter_count-1))
                    # model.soft_copy_grad(model3, iter / (iter_count - 1))
                    optimizer.step()
                    # elif method == 21:
                    #     model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=True,
                    #                                                        accumulate_gradients=False,
                    #                                                        gradient_only_modification=False)
                    if print_cuda_memory_for_some_methods:
                        print('3. ' + str(torch.cuda.memory_allocated()))
                    # !!!!Largest cuda memory consumption
                    # with open("MemoryLog.txt", "a") as myfile:
                    #     myfile.write(str(torch.cuda.memory_allocated())+',\n')

                    #loss_to_del = loss
                    #loss1_to_del = loss1
                    # loss = loss.item()
                    # loss1 = loss1.item()
                    del loss, loss1
                    #del loss_to_del, loss1_to_del
                    del output, output1
                    model.input_output_cleanup()
                    if print_cuda_memory_for_some_methods:
                        print('3.2. ' + str(torch.cuda.memory_allocated()))
                    #break

                model3.erase_grad() #optional


                #loss_ag = None
                with torch.no_grad():
                    output2 = model(data)
                    # if logit_mask is not None:
                    #     output2 += logit_mask
                    # loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)
                    loss_ag = F.cross_entropy(output2 if logit_mask is None else output2 + logit_mask, target)
                    if print_cuda_memory_for_some_methods:
                        print('4. ' + str(torch.cuda.memory_allocated()))

                loss_for_display_stats=loss_ag
                # if iter_count>2:
                if prev_iter_loss < loss_for_display_stats:  # higher loss
                    model.gradient_factor2 = model.gradient_factor2 / (
                            gradient_factor2_mul ** gradient_factor2_decrease_power)
                    if verbose and iter_count > 2:
                        verbose_str += 'H'
                elif prev_iter_loss > loss_for_display_stats:  # lower loss
                    model.gradient_factor2 = model.gradient_factor2 * gradient_factor2_mul
                    if verbose and iter_count > 2:
                        verbose_str += 'L'
                model.gradient_factor2 = min(max(model.gradient_factor2, epsilon), max_gradient_factor2)


                # if loss_backprop < loss_ag:
                #     model.gradient_factor2=model.gradient_factor2


                batch_counter += 1
                if loss_backprop < loss_ag:
                    higher_loss_batch_counter += 1
                    # #high_loss = True
                    # model.gradient_factor2 = model.gradient_factor2 / (
                    #             gradient_factor2_mul ** gradient_factor2_decrease_power)
                    # #model.gradient_factor2=min(max(model.gradient_factor2,epsilon),1.)
                    if verbose:
                        verbose_str += ' -> H'
                elif loss_backprop > loss_ag:
                    lower_loss_batch_counter += 1
                    # model.gradient_factor2 = model.gradient_factor2*gradient_factor2_mul
                    if verbose:
                        verbose_str += ' -> L'
                # model.gradient_factor2 = min(max(model.gradient_factor2, epsilon), max_gradient_factor2)

                loss_improvement += float(loss_backprop - loss_ag)

                # if float(loss - loss1) != 0.:
                #     relative_loss_improvement += float((loss - loss2) / abs(loss - loss1))
                relative_loss_improvement_denominator += float(loss_backprop - loss_initial)

                total_loss_improvement += float(loss_backprop - loss_ag)
                total_relative_loss_improvement_denominator += float(loss_backprop - loss_initial)

                if batch_idx % log_interval == 0:
                    print('GradientFactor2 = ' + str(model.gradient_factor2)+('' if not verbose else '  '+verbose_str))
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss_initial))
                    if batch_counter != 0:
                        print("Higher loss batches: " + str(higher_loss_batch_counter) + "/" + str(
                            batch_counter) + "=" + str(
                            higher_loss_batch_counter / batch_counter) + " Lower loss batch ratio: " + str(
                            lower_loss_batch_counter / batch_counter) + " Avg loss improvement: " + str(
                            loss_improvement / batch_counter) + " Avg relative loss improvement: " + str(
                            (loss_improvement / abs(
                                relative_loss_improvement_denominator)) if relative_loss_improvement_denominator != 0 else "Division by zero") + " Total avg relative loss improvement: " + str(
                            total_loss_improvement / abs(
                                total_relative_loss_improvement_denominator) if total_relative_loss_improvement_denominator != 0 else "Division by zero"))

                del output2, loss_ag
                model.input_output_cleanup()
                # print('4.2. ' + str(torch.cuda.memory_allocated()))
                continue
            elif False and method==22 and (iter_count>2 or hasattr(model,'gradient_factor2')):
                if mapping is not None:
                    target = mapping[target]

                if iter_count > 2:
                    model3.load_state_dict(model.state_dict())

                model2.load_state_dict(model.state_dict())

                # if with_optimizer_parameter_copy:
                #     opt2.load_state_dict(optimizer.state_dict())

                model2.set_require_grad(False)
                # torch.cuda.empty_cache()
                if print_cuda_memory_for_some_methods:
                    print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)
                # print('6. ' + str(torch.cuda.memory_allocated()))
                # if logit_mask is not None:
                #     output = output + logit_mask
                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                # with torch.no_grad():
                #     correct += torch.count_nonzero(target == torch.argmax(output, 1), 0)
                #     predicted += target.shape[0]

                # model2.set_require_grad(main_skip_connection_support.average_gradient_of_nonlinear_layers_enhancement)
                #opt2.load_state_dict(optimizer.state_dict())
                opt2.zero_grad()
                loss.backward(retain_graph=False)

                #########################################
                model3.copy_grad_from(model2)
                # if i == 0:
                #     opt2.zero_grad()
                #     loss.backward(retain_graph=False)
                #
                # else:
                #     model2.copy_grad_from(model)

                # print('7. ' + str(torch.cuda.memory_allocated()))
                opt2.step()

                del loss
                del output

                loss_backprop = -torch.inf
                # if main_skip_connection_support.average_gradient_of_nonlinear_layers_enhancement:
                #     model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()
                for iter in range(1, iter_count):
                    if iter != 1:
                        model2.load_state_dict(model.state_dict())
                        model.load_state_dict(model3.state_dict())

                    output1 = model(data)
                    # output1+=logit_mask
                    # if logit_mask is not None:
                    #     output1+=logit_mask
                    # loss1 = torch.nn.functional.cross_entropy(output1,
                    #                                           target)  # F.nll_loss(output1, target)
                    loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                    # loss1.requires_grad=True

                    model2.set_require_grad(True)
                    output = model2(data)
                    # output += logit_mask
                    # if logit_mask is not None:
                    #     output+=logit_mask
                    # loss = torch.nn.functional.cross_entropy(output,
                    #                                          target)  # F.nll_loss(output, target)#todo check what this line changes
                    loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                    # loss.requires_grad = True
                    if iter == 1:
                        loss_backprop = loss.item()

                    if average_gradient_of_nonlinear_layers_enhancement and not average_gradient_of_linear_layers_enhancement:
                        model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()
                    # if target.is_cuda:
                    #     print(torch.cuda.memory_allocated())
                    #     torch.cuda.empty_cache()
                    #     print(torch.cuda.memory_allocated())
                    # else:
                    #
                    #     process = psutil.Process()
                    #     print(process.memory_info().rss)  # in bytes
                    #     gc.collect()
                    #     print(process.memory_info().rss)  # in bytes

                    # print('1. ' + str(torch.cuda.memory_allocated()))
                    if method == 22:
                        if iter_count>2:
                            optimizer.param_groups[0]['alpha']=opt2.param_groups[0]['alpha']**(1./(iter_count-1))

                        #optimizer.load_state_dict(opt2.state_dict())
                        # optimizer.zero_grad()
                        model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                           accumulate_gradients=False,
                                                                           gradient_only_modification=False)
                        # todo delete all not necessary grad tensors in backward_grad_correction_with_weight_change2

                        # print('2. ' + str(torch.cuda.memory_allocated()))
                        if hasattr(model,'gradient_factor2') and model.gradient_factor2!=1.:
                            model.soft_copy_grad(model3,1.-model.gradient_factor2)
                            if iter<iter_count-1:
                                model3.copy_grad_from(model)
                        #model.soft_copy_grad(model3, 1. - iter / iter_count)
                        # model.soft_copy_grad(model3,1.-iter/(iter_count-1))
                        # model.soft_copy_grad(model3, iter / (iter_count - 1))
                        optimizer.step()
                    elif method == 21:
                        model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=True,
                                                                           accumulate_gradients=False,
                                                                           gradient_only_modification=False)
                    if print_cuda_memory_for_some_methods:
                        print('3. ' + str(torch.cuda.memory_allocated()))
                    # !!!!Largest cuda memory consumption
                    # with open("MemoryLog.txt", "a") as myfile:
                    #     myfile.write(str(torch.cuda.memory_allocated())+',\n')

                    #loss_to_del = loss
                    #loss1_to_del = loss1
                    # loss = loss.item()
                    # loss1 = loss1.item()
                    del loss, loss1
                    #del loss_to_del, loss1_to_del
                    del output, output1
                    model.input_output_cleanup()
                    if print_cuda_memory_for_some_methods:
                        print('3.2. ' + str(torch.cuda.memory_allocated()))
                    #break

                model3.erase_grad() #optional

                loss_ag = None
                with torch.no_grad():
                    output2 = model(data)
                    # if logit_mask is not None:
                    #     output2 += logit_mask
                    # loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)
                    loss_ag = F.cross_entropy(output2 if logit_mask is None else output2 + logit_mask, target)
                    if print_cuda_memory_for_some_methods:
                        print('4. ' + str(torch.cuda.memory_allocated()))

                batch_counter += 1
                if loss_backprop < loss_ag:
                    higher_loss_batch_counter += 1
                    high_loss = True
                elif loss_backprop > loss_ag:
                    lower_loss_batch_counter += 1
                loss_improvement += float(loss_backprop - loss_ag)

                # if float(loss - loss1) != 0.:
                #     relative_loss_improvement += float((loss - loss2) / abs(loss - loss1))
                relative_loss_improvement_denominator += float(loss_backprop - loss_initial)

                total_loss_improvement += float(loss_backprop - loss_ag)
                total_relative_loss_improvement_denominator += float(loss_backprop - loss_initial)

                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss_initial))
                    if batch_counter != 0:
                        print("Higher loss batches: " + str(higher_loss_batch_counter) + "/" + str(
                            batch_counter) + "=" + str(
                            higher_loss_batch_counter / batch_counter) + " Lower loss batch ratio: " + str(
                            lower_loss_batch_counter / batch_counter) + " Avg loss improvement: " + str(
                            loss_improvement / batch_counter) + " Avg relative loss improvement: " + str(
                            (loss_improvement / abs(
                                relative_loss_improvement_denominator)) if relative_loss_improvement_denominator != 0 else "Division by zero") + " Total avg relative loss improvement: " + str(
                            total_loss_improvement / abs(
                                total_relative_loss_improvement_denominator) if total_relative_loss_improvement_denominator != 0 else "Division by zero"))

                del output2, loss_ag
                model.input_output_cleanup()
                # print('4.2. ' + str(torch.cuda.memory_allocated()))
                continue
            elif method == 21 and iter_count>2:
                if mapping is not None:
                    target = mapping[target]

                if iter_count>2:
                    model3.load_state_dict(model.state_dict())

                model2.load_state_dict(model.state_dict())

                # if with_optimizer_parameter_copy:
                #     opt2.load_state_dict(optimizer.state_dict())

                model2.set_require_grad(False)
                # torch.cuda.empty_cache()
                #print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)
                #print('6. ' + str(torch.cuda.memory_allocated()))
                # if logit_mask is not None:
                #     output = output + logit_mask
                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                # with torch.no_grad():
                #     correct += torch.count_nonzero(target == torch.argmax(output, 1), 0)
                #     predicted += target.shape[0]

                # model2.set_require_grad(main_skip_connection_support.average_gradient_of_nonlinear_layers_enhancement)

                opt2.zero_grad()
                loss.backward(retain_graph=False)
                # if i == 0:
                #     opt2.zero_grad()
                #     loss.backward(retain_graph=False)
                #
                # else:
                #     model2.copy_grad_from(model)

                #print('7. ' + str(torch.cuda.memory_allocated()))
                opt2.step()

                del loss
                del output

                loss_backprop=-torch.inf
                # if main_skip_connection_support.average_gradient_of_nonlinear_layers_enhancement:
                #     model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()
                for iter in range(1,iter_count):
                    if iter!=1:
                        model2.load_state_dict(model.state_dict())
                        model.load_state_dict(model3.state_dict())

                    output1 = model(data)
                    # output1+=logit_mask
                    # if logit_mask is not None:
                    #     output1+=logit_mask
                    # loss1 = torch.nn.functional.cross_entropy(output1,
                    #                                           target)  # F.nll_loss(output1, target)
                    loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                    # loss1.requires_grad=True

                    model2.set_require_grad(True)
                    output = model2(data)
                    # output += logit_mask
                    # if logit_mask is not None:
                    #     output+=logit_mask
                    # loss = torch.nn.functional.cross_entropy(output,
                    #                                          target)  # F.nll_loss(output, target)#todo check what this line changes
                    loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                    # loss.requires_grad = True
                    if iter==1:
                        loss_backprop=loss.item()

                    if average_gradient_of_nonlinear_layers_enhancement and not average_gradient_of_linear_layers_enhancement:
                        model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()
                    # if target.is_cuda:
                    #     print(torch.cuda.memory_allocated())
                    #     torch.cuda.empty_cache()
                    #     print(torch.cuda.memory_allocated())
                    # else:
                    #
                    #     process = psutil.Process()
                    #     print(process.memory_info().rss)  # in bytes
                    #     gc.collect()
                    #     print(process.memory_info().rss)  # in bytes

                    #print('1. ' + str(torch.cuda.memory_allocated()))
                    if method == 22:
                        model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                           accumulate_gradients=False,
                                                                           gradient_only_modification=False)
                        # todo delete all not necessary grad tensors in backward_grad_correction_with_weight_change2

                        #print('2. ' + str(torch.cuda.memory_allocated()))
                        optimizer.step()
                    elif method == 21:
                        model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=True,
                                                                           accumulate_gradients=False,
                                                                           gradient_only_modification=False)
                    #print('3. ' + str(torch.cuda.memory_allocated()))
                    #!!!!Largest cuda memory consumption
                    # with open("MemoryLog.txt", "a") as myfile:
                    #     myfile.write(str(torch.cuda.memory_allocated())+',\n')

                    loss_to_del = loss
                    loss1_to_del = loss1
                    #loss = loss.item()
                    #loss1 = loss1.item()
                    del loss_to_del, loss1_to_del
                    del output, output1
                    model.input_output_cleanup()
                    #print('3.2. ' + str(torch.cuda.memory_allocated()))

                loss_ag = None
                with torch.no_grad():
                    output2 = model(data)
                    # if logit_mask is not None:
                    #     output2 += logit_mask
                    # loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)
                    loss_ag = F.cross_entropy(output2 if logit_mask is None else output2 + logit_mask, target)
                    #print('4. ' + str(torch.cuda.memory_allocated()))

                batch_counter += 1
                if loss_backprop < loss_ag:
                    higher_loss_batch_counter += 1
                    high_loss = True
                elif loss_backprop > loss_ag:
                    lower_loss_batch_counter += 1
                loss_improvement += float(loss_backprop - loss_ag)

                # if float(loss - loss1) != 0.:
                #     relative_loss_improvement += float((loss - loss2) / abs(loss - loss1))
                relative_loss_improvement_denominator += float(loss_backprop - loss_initial)

                total_loss_improvement += float(loss_backprop - loss_ag)
                total_relative_loss_improvement_denominator += float(loss_backprop - loss_initial)

                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss_initial))
                    if batch_counter != 0:
                        print("Higher loss batches: " + str(higher_loss_batch_counter) + "/" + str(
                            batch_counter) + "=" + str(
                            higher_loss_batch_counter / batch_counter) + " Lower loss batch ratio: " + str(
                            lower_loss_batch_counter / batch_counter) + " Avg loss improvement: " + str(
                            loss_improvement / batch_counter) + " Avg relative loss improvement: " + str(
                            (loss_improvement / abs(
                                relative_loss_improvement_denominator)) if relative_loss_improvement_denominator != 0 else "Division by zero") + " Total avg relative loss improvement: " + str(
                            total_loss_improvement / abs(
                                total_relative_loss_improvement_denominator) if total_relative_loss_improvement_denominator != 0 else "Division by zero"))

                del output2, loss_ag
                model.input_output_cleanup()
                #print('4.2. ' + str(torch.cuda.memory_allocated()))
                continue
            elif method==21 or method==22 or (method>=24 and method<=29):

                m22_l2_norm=method==22
                grad_l2_norm=None

                # for batch_idx, (data, target) in enumerate(train_loader):
                #     data, target = data.to(device), target.to(device)
                if mapping is not None:
                    target = mapping[target]


                model2.load_state_dict(model.state_dict())

                if method!=28 and method!=29:
                    if method==24:
                        #copy_optimizer_params(opt2, optimizer, model2, model)
                        try:
                            copy_optimizer_params(opt2, optimizer, model2, model)
                        except:
                            opt2.load_state_dict(optimizer.state_dict().copy())
                    else:
                        try:
                            copy_optimizer_params(optimizer,opt2,model,model2)
                        except:
                            # optimizer.load_state_dict(opt2.state_dict().copy())
                            optimizer.load_state_dict(copy.deepcopy(opt2.state_dict()))

                if method==28 or method==29:
                    betas=optimizer.param_groups[0]['betas']
                    betas_zero_momentum=(0.,betas[1])
                    optimizer.param_groups[0]['betas']=betas_zero_momentum

                    if method==28:
                        opt2.param_groups[0]['lr']=0.5*optimizer.param_groups[0]['lr']#10.0*optimizer.param_groups[0]['lr']


                # if with_optimizer_parameter_copy:
                #     opt2.load_state_dict(optimizer.state_dict())
                #opt2.load_state_dict(optimizer)


                model2.set_require_grad(False)
                # torch.cuda.empty_cache()
                #print('5. ' + str(torch.cuda.memory_allocated()))
                output = model2(data)
                #print('6. ' + str(torch.cuda.memory_allocated()))
                # if logit_mask is not None:
                #     output = output + logit_mask
                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True
                loss_initial = loss.item()

                # with torch.no_grad():
                #     correct += torch.count_nonzero(target == torch.argmax(output, 1), 0)
                #     predicted += target.shape[0]

                # model2.set_require_grad(main_skip_connection_support.average_gradient_of_nonlinear_layers_enhancement)

                opt2.zero_grad()
                loss.backward(retain_graph=False)

                if m22_l2_norm:
                    grad_l2_norm=model2.grad_l2_norm()
                # if i == 0:
                #     opt2.zero_grad()
                #     loss.backward(retain_graph=False)
                #
                # else:
                #     model2.copy_grad_from(model)

                #print('7. ' + str(torch.cuda.memory_allocated()))
                opt2.step()

                del loss
                del output

                # if main_skip_connection_support.average_gradient_of_nonlinear_layers_enhancement:
                #     model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()

                output1 = model(data)
                # output1+=logit_mask
                # if logit_mask is not None:
                #     output1+=logit_mask
                # loss1 = torch.nn.functional.cross_entropy(output1,
                #                                           target)  # F.nll_loss(output1, target)
                loss1 = F.cross_entropy(output1 if logit_mask is None else output1 + logit_mask, target)
                # loss1.requires_grad=True

                model2.set_require_grad(True)
                output = model2(data)
                # output += logit_mask
                # if logit_mask is not None:
                #     output+=logit_mask
                # loss = torch.nn.functional.cross_entropy(output,
                #                                          target)  # F.nll_loss(output, target)#todo check what this line changes
                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)
                # loss.requires_grad = True

                if average_gradient_of_nonlinear_layers_enhancement and not average_gradient_of_linear_layers_enhancement:
                    model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()
                # if target.is_cuda:
                #     print(torch.cuda.memory_allocated())
                #     torch.cuda.empty_cache()
                #     print(torch.cuda.memory_allocated())
                # else:
                #
                #     process = psutil.Process()
                #     print(process.memory_info().rss)  # in bytes
                #     gc.collect()
                #     print(process.memory_info().rss)  # in bytes

                #print('1. ' + str(torch.cuda.memory_allocated()))
                if (method >= 22 and method<=24):
                    if step_length_stats:
                        _step_length_base_l1=model.step_length(model2, L=1)
                        _step_length_base_l2 = model.step_length(model2, L=2)
                        step_length_base_l1 +=_step_length_base_l1
                        step_length_base_l2 += _step_length_base_l2

                    model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                       accumulate_gradients=False,
                                                                       gradient_only_modification=False)
                    # todo delete all not necessary grad tensors in backward_grad_correction_with_weight_change2

                    #print('2. ' + str(torch.cuda.memory_allocated()))
                    #model.avg_grad_to_integrated_grad(model2,model)
                    if m22_l2_norm:
                        avg_grad_l2_norm = model.grad_l2_norm()
                        if avg_grad_l2_norm!=0.:
                            model.mul_grad(grad_l2_norm/avg_grad_l2_norm)

                    if step_length_stats:
                        model2.load_state_dict(model.state_dict())

                    optimizer.step()
                    if method==24:
                        model.load_state_dict(model2.state_dict())

                    if step_length_stats:
                        step_length_l1 += model.step_length(model2, L=1)
                        step_length_l2 += model.step_length(model2, L=2)

                        if print_step_length:
                            print('Method 22 step length L2: '+str(100.*step_length_l2/step_length_base_l2 if step_length_base_l2!=0. else 'Inf')+'%  L1: '+str(100.*step_length_l1/step_length_base_l1 if step_length_base_l1!=0. else 'Inf')+'%')

                elif method == 21 or method==25:
                    if method==21:
                        model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=True,
                                                                           accumulate_gradients=False,
                                                                           gradient_only_modification=False)
                    else:
                        model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                           accumulate_gradients=False,
                                                                           gradient_only_modification=False)
                        model.weight_change_as_model_but_directed_by_grad(model2)
                elif method==28 or method==29:
                    model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                       accumulate_gradients=False,
                                                                       gradient_only_modification=False)
                    optimizer.step()
                else:
                    #method 26 or 27
                    if step_length_stats:
                        # step_length_base_l1 += model.step_length(model2, L=1)
                        # step_length_base_l2 += model.step_length(model2, L=2)
                        _step_length_base_l1 = model.step_length(model2, L=1)
                        _step_length_base_l2 = model.step_length(model2, L=2)
                        step_length_base_l1 += _step_length_base_l1
                        step_length_base_l2 += _step_length_base_l2

                    model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                       accumulate_gradients=False,
                                                                       gradient_only_modification=False)
                    model.layerwise_grad_normalization(model2,L=2)
                    if step_length_stats:
                        model2.load_state_dict(model.state_dict())

                    optimizer.step()

                    if step_length_stats:
                        if method==27:
                            _step_length_l1 = model.step_length(model2, L=1)
                            _step_length_l2 = model.step_length(model2, L=2)
                            if _step_length_l1!=0:
                                model.mul_update(model2,_step_length_base_l1/_step_length_l1)
                            # if _step_length_l2!=0 and _step_length_l1!=0:
                            #     model.mul_update(model2,_step_length_base_l2/_step_length_l2)
                            # step_length_l1 += _step_length_l1
                            # step_length_l2 += _step_length_l2
                            step_length_l1 += model.step_length(model2, L=1)
                            step_length_l2 += model.step_length(model2, L=2)
                        else:
                            step_length_l1 = model.step_length(model2, L=1)
                            step_length_l2 = model.step_length(model2, L=2)


                        if print_step_length:
                            print('Method 26 step length L2: '+str(100.*step_length_l2/step_length_base_l2 if step_length_base_l2!=0. else 'Inf')+'%  L1: '+str(100.*step_length_l1/step_length_base_l1 if step_length_base_l1!=0. else 'Inf')+'%')
                #print('3. ' + str(torch.cuda.memory_allocated()))
                #!!!!Largest cuda memory consumption
                # with open("MemoryLog.txt", "a") as myfile:
                #     myfile.write(str(torch.cuda.memory_allocated())+',\n')

                loss_to_del = loss
                loss1_to_del = loss1
                loss = loss.item()
                loss1 = loss1.item()
                del loss_to_del, loss1_to_del
                del output, output1
                model.input_output_cleanup()
                #print('3.2. ' + str(torch.cuda.memory_allocated()))

                loss2 = None
                with torch.no_grad():
                    output2 = model(data)
                    # if logit_mask is not None:
                    #     output2 += logit_mask
                    # loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)
                    loss2 = F.cross_entropy(output2 if logit_mask is None else output2 + logit_mask, target)
                    #print('4. ' + str(torch.cuda.memory_allocated()))

                batch_counter += 1
                if loss < loss2:
                    higher_loss_batch_counter += 1
                    high_loss = True
                    #model.load_state_dict(model2.state_dict())
                elif loss > loss2:
                    lower_loss_batch_counter += 1
                loss_improvement += float(loss - loss2)

                # if float(loss - loss1) != 0.:
                #     relative_loss_improvement += float((loss - loss2) / abs(loss - loss1))
                relative_loss_improvement_denominator += float(loss - loss1)

                total_loss_improvement += float(loss - loss2)
                total_relative_loss_improvement_denominator += float(loss - loss1)

                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss_initial))
                    if batch_counter != 0:
                        print("Higher loss batches: " + str(higher_loss_batch_counter) + "/" + str(
                            batch_counter) + "=" + str(
                            higher_loss_batch_counter / batch_counter) + " Lower loss batch ratio: " + str(
                            lower_loss_batch_counter / batch_counter) + " Avg loss improvement: " + str(
                            loss_improvement / batch_counter) + " Avg relative loss improvement: " + str(
                            (loss_improvement / abs(
                                relative_loss_improvement_denominator)) if relative_loss_improvement_denominator != 0 else "Division by zero") + " Total avg relative loss improvement: " + str(
                            total_loss_improvement / abs(
                                total_relative_loss_improvement_denominator) if total_relative_loss_improvement_denominator != 0 else "Division by zero"))

                del output2, loss2
                model.input_output_cleanup()
                #print('4.2. ' + str(torch.cuda.memory_allocated()))
                continue
            elif method==20:
                if mapping is not None:
                    target = mapping[target]

                optimizer.zero_grad()

                output = model(data)
                # if logit_mask is not None:
                #     output += logit_mask
                # loss = F.cross_entropy(output, target)  # F.nll_loss(output, target)
                loss = F.cross_entropy(output if logit_mask is None else output + logit_mask, target)

                loss.backward()
                optimizer.step()
                if show_grad:
                    print(model.layers[0].weight.grad[0][0][0])
            elif method==11 or method==12:

                if iter_count>=3:

                    if not initialized:
                        initialized=True
                        #opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))

                        out = model(data)
                        out = torch.sum(out)
                        zero = out - out
                        zero.backward()
                        optimizer.step()
                        optimizer.param_groups[0][momentum_key] = zero_momentum
                        for param, val in optimizer.state.items():
                            if 'step' in val:
                                optimizer.state[param]['step'] = optimizer.state[param]['step'] - 1
                        # optimizer.load_state_dict(copy.deepcopy(opt2.state_dict()))
                        opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                        opt2.param_groups[0]['lr'] = lr_including_momentum
                        opt2.param_groups[0][momentum_key] = zero_momentum
                        if opt3 is not None:
                            opt3.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                            opt3.param_groups[0]['lr'] = lr_including_momentum
                            opt3.param_groups[0][momentum_key] = zero_momentum

                    optimizer.param_groups[0]['lr'] = lr_including_momentum
                    optimizer.param_groups[0][momentum_key] = zero_momentum



                    intermediate_updates_by_RMSprop = False
                    additional_enhancement=not intermediate_updates_by_RMSprop and method==9

                    model2.load_state_dict(model.state_dict())
                    model3.load_state_dict(model.state_dict())
                    # if intermediate_updates_by_RMSprop:
                    #     opt3.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                    opt3.load_state_dict(copy.deepcopy(optimizer.state_dict()))


                    output = model2(data)
                    _loss = F.cross_entropy(output, target)  # F.nll_loss(output, target)

                    if intermediate_updates_by_RMSprop:
                        opt2.load_state_dict(copy.deepcopy(opt3.state_dict()))

                    opt2.zero_grad()
                    _loss.backward()

                    opt2.step()
                    _output = model2(data)
                    loss_backprop=F.cross_entropy(_output, target)

                    if not intermediate_updates_by_RMSprop:
                        model3.copy_grad_from(model2)

                    for i in range(iter_count-1):
                        if i%2==0:
                            if i!=0:
                                model.load_state_dict(model3.state_dict())
                                _output = model2(data)
                                loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)
                            else:
                                loss=loss_backprop

                            output1 = model(data)
                            loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)

                            if i==iter_count-2:
                                if not intermediate_updates_by_RMSprop and not additional_enhancement:
                                    model2.copy_grad_from(model3)

                                if intermediate_updates_by_RMSprop:
                                    optimizer.load_state_dict(copy.deepcopy(opt3.state_dict()))
                                model.backward_grad_correction_with_weight_change2(loss1, model2,weight_change=False,gradient_only_modification=True)

                                optimizer.param_groups[0]['lr']=lr_initial
                                optimizer.param_groups[0][momentum_key]=optim_args[momentum_key]

                                optimizer.step()
                            else:
                                if intermediate_updates_by_RMSprop:
                                    optimizer.load_state_dict(copy.deepcopy(opt3.state_dict()))
                                    model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                                       gradient_only_modification=True)
                                    optimizer.step()
                                else:
                                    model.backward_grad_correction_with_weight_change2(loss1, model2)
                        else:
                            model2.load_state_dict(model3.state_dict())

                            _output = model(data)
                            loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)

                            output1 = model2(data)
                            loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)

                            if i == iter_count - 2:
                                #model.copy_grad_from(model2)
                                if not intermediate_updates_by_RMSprop and not additional_enhancement:
                                    model.copy_grad_from(model3)
                                model2.backward_grad_correction_with_weight_change2(loss1, model,weight_change=False,gradient_only_modification=True)
                            else:
                                if intermediate_updates_by_RMSprop:
                                    opt2.load_state_dict(copy.deepcopy(opt3.state_dict()))
                                    model2.backward_grad_correction_with_weight_change2(loss1, model, weight_change=False,
                                                                                        gradient_only_modification=True)
                                    opt2.step()
                                else:
                                    model2.backward_grad_correction_with_weight_change2(loss1, model)

                    if (iter_count)%2==1:
                        if intermediate_updates_by_RMSprop:
                            optimizer.load_state_dict(copy.deepcopy(opt3.state_dict()))
                        #model.load_state_dict(model2.state_dict())
                        model.load_state_dict(model3.state_dict())
                        if not additional_enhancement:
                            model.copy_grad_from(model2)

                        optimizer.param_groups[0]['lr'] = lr_initial
                        optimizer.param_groups[0][momentum_key] = optim_args[momentum_key]

                        optimizer.step()

                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)

                    batch_counter += 1
                    if loss_backprop < loss2:
                        higher_loss_batch_counter += 1
                        # high_loss=True
                    elif loss_backprop > loss2:
                        lower_loss_batch_counter += 1
                    loss_improvement += float(loss_backprop.item() - loss2.item())

                    relative_loss_improvement_denominator += float(loss_backprop.item() - _loss.item())

                    total_loss_improvement += float(loss_backprop.item() - loss2.item())
                    total_relative_loss_improvement_denominator += float(loss_backprop.item() - _loss.item())
                else:
                    if not initialized:
                        initialized = True

                        out = model2(data)
                        out = torch.sum(out)
                        zero = out - out
                        zero.backward()
                        opt2.step()
                        opt2.param_groups[0][momentum_key] = zero_momentum
                        for param, val in opt2.state.items():
                            if 'step' in val:
                                opt2.state[param]['step'] = opt2.state[param]['step'] - 1
                        # optimizer.load_state_dict(copy.deepcopy(opt2.state_dict()))
                        # opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                        # if opt3 is not None:
                        #     opt3.load_state_dict(copy.deepcopy(optimizer.state_dict()))

                        opt2.param_groups[0]['lr']=lr_including_momentum
                        optimizer.param_groups[0]['lr']=lr_initial


                    model2.load_state_dict(model.state_dict())

                    output = model2(data)
                    _loss = F.cross_entropy(output, target)#F.nll_loss(output, target)

                    opt2.zero_grad()

                    if average_gradient_of_linear_layers_enhancement:
                        model2.backward(_loss)
                    else:
                        _loss.backward()

                    opt2.step()

                    if show_grad:
                        print(model2.layers[0].weight.grad[0][0][0])

                    _output = model2(data)
                    loss = F.cross_entropy(_output, target)#F.nll_loss(output, target)#todo check what this line changes

                    output1 = model(data)
                    loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)
                    #model.backward_grad_correction(loss1, model2, F.nll_loss, target)
                    #optimizer.zero_grad()
                    model.backward_grad_correction_with_weight_change2(loss1, model2,weight_change=False,gradient_only_modification=True)
                    optimizer.step()

                    if show_grad:
                        print(model.layers[0].weight.grad[0][0][0])
                    # if common_optimizer_parameters_enhancement:
                    #     #model.backward_grad_correction_with_weight_change2(loss1, model2, loss, F.cross_entropy, target)
                    #     model2.copy_grad_from(model)
                    #     opt2.step()

                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)#F.nll_loss(output2, target)

                    batch_counter+=1
                    if loss<loss2:
                        higher_loss_batch_counter+=1
                        #high_loss=True
                        if take_optimal_update:
                            model.load_state_dict(model2.state_dict())
                    elif loss>loss2:
                        lower_loss_batch_counter+=1
                    loss_improvement+=float(loss.item()-loss2.item())

                    relative_loss_improvement_denominator += float(loss.item() - _loss.item())

                    total_loss_improvement += float(loss.item()-loss2.item())
                    total_relative_loss_improvement_denominator += float(loss.item() - _loss.item())
                loss=_loss
            elif method==10:
                if not initialized:
                    initialized = True

                    out=model(data)
                    out=torch.sum(out)
                    zero=out-out
                    zero.backward()
                    optimizer.step()
                    optimizer.param_groups[0]['momentum'] = 0.
                    for param, val in optimizer.state.items():
                        if 'step' in val:
                            optimizer.state[param]['step'] = optimizer.state[param]['step']-1
                    #optimizer.load_state_dict(copy.deepcopy(opt2.state_dict()))
                    opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                    if opt3 is not None:
                        opt3.load_state_dict(copy.deepcopy(optimizer.state_dict()))

                opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                model2.load_state_dict(model.state_dict())
                opt2.zero_grad()
                output2=model2(data)
                loss2=F.cross_entropy(output2, target)
                loss2.backward()
                opt2.step()
                # for param, val in optimizer.state.items():
                #     if 'square_avg' in val:
                #         opt2.state[param]['momentum_buffer']=optim_args['momentum']*opt2.state[param]['momentum_buffer']+(1.-optim_args['momentum'])*
                #         #optimizer.state[param]['step'] = optimizer.state[param]['step'] + 1
                with torch.no_grad():
                    for i in range(len(model.layers)):
                        if hasattr(model.layers[i],'weight'):
                            params=model.layers[i].weight
                            params2=model2.layers[i].weight
                            opt2.state[params2]['momentum_buffer']=optim_args['momentum']*opt2.state[params2]['momentum_buffer']+(1.-optim_args['momentum'])*(params2-params)
                            params+=opt2.state[params2]['momentum_buffer']

                            params = model.layers[i].bias
                            params2 = model2.layers[i].bias
                            opt2.state[params2]['momentum_buffer'] = optim_args['momentum'] * opt2.state[params2][
                                'momentum_buffer'] + (1. - optim_args['momentum']) * (params2 - params)
                            params += opt2.state[params2]['momentum_buffer']
                loss=loss2
                # ...
                # opt2.state['step'] = copy.deepcopy(optimizer.state['step'])
                # for param, val in optimizer.state.items():
                #     #print(param)
                #     if 'exp_avg_sq' in val:
                #         opt2.state[param]['step'] = copy.deepcopy(val['step'])
                #         opt2.state[param]['square_avg'] = copy.deepcopy(val['exp_avg_sq'])
                # ...
            elif method==8 or method==9:
                # optimizer.param_groups[0]['lr'] = lr_including_momentum
                # opt2.param_groups[0]['lr'] = lr_including_momentum
                #
                # optimizer.param_groups[0]['betas']=(0.,betas[1])
                # opt2.param_groups[0]['betas']=(0.,betas[1])

                #optimizer.load_state_dict(copy.deepcopy(opt2.state_dict()))
                #optimizer.load_state_dict(opt2.state_dict())
                #torch.manual_seed(1)
                #optimizer.load_state_dict(copy.deepcopy(opt2.state_dict()))



                if iter_count>=3:
                    intermediate_updates_by_RMSprop = False
                    additional_enhancement=not intermediate_updates_by_RMSprop and method==9

                    model2.load_state_dict(model.state_dict())
                    model3.load_state_dict(model.state_dict())
                    if intermediate_updates_by_RMSprop:
                        opt3.load_state_dict(copy.deepcopy(optimizer.state_dict()))


                    output = model2(data)
                    _loss = F.cross_entropy(output, target)  # F.nll_loss(output, target)

                    if intermediate_updates_by_RMSprop:
                        opt2.load_state_dict(copy.deepcopy(opt3.state_dict()))

                    opt2.zero_grad()
                    _loss.backward()

                    opt2.step()
                    _output = model2(data)
                    loss_backprop=F.cross_entropy(_output, target)

                    if not intermediate_updates_by_RMSprop:
                        model3.copy_grad_from(model2)

                    for i in range(iter_count-1):
                        if i%2==0:
                            if i!=0:
                                model.load_state_dict(model3.state_dict())
                                _output = model2(data)
                                loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)
                            else:
                                loss=loss_backprop

                            output1 = model(data)
                            loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)

                            if i==iter_count-2:
                                if not intermediate_updates_by_RMSprop and not additional_enhancement:
                                    model2.copy_grad_from(model3)

                                if intermediate_updates_by_RMSprop:
                                    optimizer.load_state_dict(copy.deepcopy(opt3.state_dict()))
                                model.backward_grad_correction_with_weight_change2(loss1, model2,weight_change=False,gradient_only_modification=True)
                                optimizer.step()
                            else:
                                if intermediate_updates_by_RMSprop:
                                    optimizer.load_state_dict(copy.deepcopy(opt3.state_dict()))
                                    model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                                       gradient_only_modification=True)
                                    optimizer.step()
                                else:
                                    model.backward_grad_correction_with_weight_change2(loss1, model2)
                        else:
                            model2.load_state_dict(model3.state_dict())

                            _output = model(data)
                            loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)

                            output1 = model2(data)
                            loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)

                            if i == iter_count - 2:
                                #model.copy_grad_from(model2)
                                if not intermediate_updates_by_RMSprop and not additional_enhancement:
                                    model.copy_grad_from(model3)
                                model2.backward_grad_correction_with_weight_change2(loss1, model,weight_change=False,gradient_only_modification=True)
                            else:
                                if intermediate_updates_by_RMSprop:
                                    opt2.load_state_dict(copy.deepcopy(opt3.state_dict()))
                                    model2.backward_grad_correction_with_weight_change2(loss1, model, weight_change=False,
                                                                                        gradient_only_modification=True)
                                    opt2.step()
                                else:
                                    model2.backward_grad_correction_with_weight_change2(loss1, model)

                    if (iter_count)%2==1:
                        if intermediate_updates_by_RMSprop:
                            optimizer.load_state_dict(copy.deepcopy(opt3.state_dict()))
                        #model.load_state_dict(model2.state_dict())
                        model.load_state_dict(model3.state_dict())
                        if not additional_enhancement:
                            model.copy_grad_from(model2)
                        optimizer.step()

                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)

                    batch_counter += 1
                    if loss_backprop < loss2:
                        higher_loss_batch_counter += 1
                        # high_loss=True
                    elif loss_backprop > loss2:
                        lower_loss_batch_counter += 1
                    loss_improvement += float(loss_backprop.item() - loss2.item())

                    relative_loss_improvement_denominator += float(loss_backprop.item() - _loss.item())

                    total_loss_improvement += float(loss_backprop.item() - loss2.item())
                    total_relative_loss_improvement_denominator += float(loss_backprop.item() - _loss.item())
                else:
                    model2.load_state_dict(model.state_dict())

                    output = model2(data)
                    _loss = F.cross_entropy(output, target)#F.nll_loss(output, target)

                    opt2.zero_grad()

                    if average_gradient_of_linear_layers_enhancement:
                        model2.backward(_loss)
                    else:
                        _loss.backward()

                    opt2.step()

                    if show_grad:
                        print(model2.layers[0].weight.grad[0][0][0])

                    _output = model2(data)
                    loss = F.cross_entropy(_output, target)#F.nll_loss(output, target)#todo check what this line changes

                    output1 = model(data)
                    loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)
                    #model.backward_grad_correction(loss1, model2, F.nll_loss, target)
                    #optimizer.zero_grad()
                    model.backward_grad_correction_with_weight_change2(loss1, model2,weight_change=False,gradient_only_modification=True)
                    optimizer.step()

                    if show_grad:
                        print(model.layers[0].weight.grad[0][0][0])
                    # if common_optimizer_parameters_enhancement:
                    #     #model.backward_grad_correction_with_weight_change2(loss1, model2, loss, F.cross_entropy, target)
                    #     model2.copy_grad_from(model)
                    #     opt2.step()

                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)#F.nll_loss(output2, target)

                    batch_counter+=1
                    if loss<loss2:
                        higher_loss_batch_counter+=1
                        #high_loss=True
                        if take_optimal_update:
                            model.load_state_dict(model2.state_dict())
                    elif loss>loss2:
                        lower_loss_batch_counter+=1
                    loss_improvement+=float(loss.item()-loss2.item())

                    relative_loss_improvement_denominator += float(loss.item() - _loss.item())

                    total_loss_improvement += float(loss.item()-loss2.item())
                    total_relative_loss_improvement_denominator += float(loss.item() - _loss.item())

                # optimizer.param_groups[0]['lr'] = lr_initial
                # opt2.param_groups[0]['lr'] = lr_initial
                #
                # optimizer.param_groups[0]['betas'] = betas
                # opt2.param_groups[0]['betas'] = betas
            elif method==7:
                if iter_count>=3:
                    model2.load_state_dict(model.state_dict())
                    model3.load_state_dict(model.state_dict())

                    output = model2(data)
                    _loss = F.cross_entropy(output, target)  # F.nll_loss(output, target)

                    opt2.zero_grad()
                    _loss.backward()
                    # if i == 0:
                    #     opt2.zero_grad()
                    #     _loss.backward()
                    #     # model2.backward(_loss)
                    #     # loss.backward(retain_graph=True)
                    # else:
                    #     model2.copy_grad_from(model)

                    opt2.step()
                    _output = model2(data)
                    loss_backprop=F.cross_entropy(_output, target)

                    for i in range(iter_count-1):
                        if i%2==0:
                            if i!=0:
                                model.load_state_dict(model3.state_dict())
                                _output = model2(data)
                                loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)
                            else:
                                loss=loss_backprop

                            output1 = model(data)
                            loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)

                            model.backward_grad_correction_with_weight_change2_parametrized_relative_step_size_and_no_negations_in_step_direction(loss1, model2,lr_mul=1./method_7_lr_mul if iter_count-2==i else 1.)
                        else:
                            model2.load_state_dict(model3.state_dict())

                            _output = model(data)
                            loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)

                            output1 = model2(data)
                            loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)

                            model2.backward_grad_correction_with_weight_change2_parametrized_relative_step_size_and_no_negations_in_step_direction(loss1, model,lr_mul=1./method_7_lr_mul if iter_count-2==i else 1.)

                    if (iter_count)%2==1:
                        model.load_state_dict(model2.state_dict())

                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)

                    batch_counter += 1
                    if loss_backprop < loss2:
                        higher_loss_batch_counter += 1
                        # high_loss=True
                    elif loss_backprop > loss2:
                        lower_loss_batch_counter += 1
                    loss_improvement += float(loss_backprop.item() - loss2.item())

                    # if float(loss_backprop - _loss) != 0.:
                    #     relative_loss_improvement += float((loss_backprop - loss2) / abs(loss_backprop - _loss))
                    relative_loss_improvement_denominator += float(loss_backprop.item() - _loss.item())

                    total_loss_improvement += float(loss_backprop.item() - loss2.item())
                    total_relative_loss_improvement_denominator += float(loss_backprop.item() - _loss.item())
                else:
                    model2.load_state_dict(model.state_dict())

                    output = model2(data)
                    _loss = F.cross_entropy(output, target)#F.nll_loss(output, target)

                    opt2.zero_grad()

                    #_loss.backward()
                    if average_gradient_of_linear_layers_enhancement:
                        model2.backward(_loss)
                    else:
                        _loss.backward()

                    opt2.step()

                    if show_grad:
                        print(model2.layers[0].weight.grad[0][0][0])

                    _output = model2(data)
                    loss = F.cross_entropy(_output, target)#F.nll_loss(output, target)#todo check what this line changes
                    #model.load_state_dict(model2.state_dict())
                    #break

                    output1 = model(data)
                    loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)
                    #model.backward_grad_correction(loss1, model2, F.nll_loss, target)
                    model.backward_grad_correction_with_weight_change2_parametrized_relative_step_size_and_no_negations_in_step_direction(loss1, model2,lr_mul=1./method_7_lr_mul)

                    if show_grad:
                        print(model.layers[0].weight.grad[0][0][0])
                    if common_optimizer_parameters_enhancement:
                        #model.backward_grad_correction_with_weight_change2(loss1, model2, loss, F.cross_entropy, target)
                        model2.copy_grad_from(model)
                        opt2.step()
                    # model2.backward_grad_correction(loss,model)

                    #optimizer.step()
                    #opt2.zero_grad()

                    if False and additional_upgrade:
                        model2.load_state_dict(model.state_dict())
                        opt2.load_state_dict(optimizer.state_dict())

                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)#F.nll_loss(output2, target)

                    batch_counter+=1
                    if loss<loss2:
                        higher_loss_batch_counter+=1
                        #high_loss=True
                        if take_optimal_update:
                            model.load_state_dict(model2.state_dict())
                    elif loss>loss2:
                        lower_loss_batch_counter+=1
                    loss_improvement+=float(loss.item()-loss2.item())

                    relative_loss_improvement_denominator += float(loss.item() - _loss.item())

                    total_loss_improvement += float(loss.item()-loss2.item())
                    total_relative_loss_improvement_denominator += float(loss.item() - _loss.item())
            elif method==6:
                if iter_count>=3:
                    model2.load_state_dict(model.state_dict())
                    model3.load_state_dict(model.state_dict())

                    output = model2(data)
                    _loss = F.cross_entropy(output, target)  # F.nll_loss(output, target)

                    opt2.zero_grad()
                    _loss.backward()
                    # if i == 0:
                    #     opt2.zero_grad()
                    #     _loss.backward()
                    #     # model2.backward(_loss)
                    #     # loss.backward(retain_graph=True)
                    # else:
                    #     model2.copy_grad_from(model)

                    opt2.step()
                    _output = model2(data)
                    loss_backprop=F.cross_entropy(_output, target)

                    for i in range(iter_count-1):
                        if i%2==0:
                            if i!=0:
                                model.load_state_dict(model3.state_dict())
                                _output = model2(data)
                                loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)
                            else:
                                loss=loss_backprop

                            output1 = model(data)
                            loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)

                            model.backward_grad_correction_with_weight_change2_parametrized_relative_step_size_and_no_negations_in_step_direction(loss1, model2)
                        else:
                            model2.load_state_dict(model3.state_dict())

                            _output = model(data)
                            loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)

                            output1 = model2(data)
                            loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)

                            model2.backward_grad_correction_with_weight_change2_parametrized_relative_step_size_and_no_negations_in_step_direction(loss1, model)

                    if (iter_count)%2==1:
                        model.load_state_dict(model2.state_dict())

                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)

                    batch_counter += 1
                    if loss_backprop < loss2:
                        higher_loss_batch_counter += 1
                        # high_loss=True
                    elif loss_backprop > loss2:
                        lower_loss_batch_counter += 1
                    loss_improvement += float(loss_backprop.item() - loss2.item())

                    # if float(loss_backprop - _loss) != 0.:
                    #     relative_loss_improvement += float((loss_backprop - loss2) / abs(loss_backprop - _loss))
                    relative_loss_improvement_denominator += float(loss_backprop.item() - _loss.item())

                    total_loss_improvement += float(loss_backprop.item() - loss2.item())
                    total_relative_loss_improvement_denominator += float(loss_backprop.item() - _loss.item())
                else:
                    model2.load_state_dict(model.state_dict())

                    output = model2(data)
                    _loss = F.cross_entropy(output, target)#F.nll_loss(output, target)

                    opt2.zero_grad()

                    #_loss.backward()
                    if average_gradient_of_linear_layers_enhancement:
                        model2.backward(_loss)
                    else:
                        _loss.backward()

                    opt2.step()

                    if show_grad:
                        print(model2.layers[0].weight.grad[0][0][0])

                    _output = model2(data)
                    loss = F.cross_entropy(_output, target)#F.nll_loss(output, target)#todo check what this line changes
                    #model.load_state_dict(model2.state_dict())
                    #break

                    output1 = model(data)
                    loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)
                    #model.backward_grad_correction(loss1, model2, F.nll_loss, target)
                    model.backward_grad_correction_with_weight_change2_parametrized_relative_step_size_and_no_negations_in_step_direction(loss1, model2)

                    if show_grad:
                        print(model.layers[0].weight.grad[0][0][0])
                    if common_optimizer_parameters_enhancement:
                        #model.backward_grad_correction_with_weight_change2(loss1, model2, loss, F.cross_entropy, target)
                        model2.copy_grad_from(model)
                        opt2.step()
                    # model2.backward_grad_correction(loss,model)

                    #optimizer.step()
                    #opt2.zero_grad()

                    if False and additional_upgrade:
                        model2.load_state_dict(model.state_dict())
                        opt2.load_state_dict(optimizer.state_dict())

                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)#F.nll_loss(output2, target)

                    batch_counter+=1
                    if loss<loss2:
                        higher_loss_batch_counter+=1
                        #high_loss=True
                        if take_optimal_update:
                            model.load_state_dict(model2.state_dict())
                    elif loss>loss2:
                        lower_loss_batch_counter+=1
                    loss_improvement+=float(loss.item()-loss2.item())

                    relative_loss_improvement_denominator += float(loss.item() - _loss.item())

                    total_loss_improvement += float(loss.item()-loss2.item())
                    total_relative_loss_improvement_denominator += float(loss.item() - _loss.item())
            elif method==5:
                if iter_count>=3:
                    model2.load_state_dict(model.state_dict())
                    model3.load_state_dict(model.state_dict())

                    output = model2(data)
                    _loss = F.cross_entropy(output, target)  # F.nll_loss(output, target)

                    opt2.zero_grad()
                    _loss.backward()
                    # if i == 0:
                    #     opt2.zero_grad()
                    #     _loss.backward()
                    #     # model2.backward(_loss)
                    #     # loss.backward(retain_graph=True)
                    # else:
                    #     model2.copy_grad_from(model)

                    opt2.step()
                    _output = model2(data)
                    loss_backprop=F.cross_entropy(_output, target)

                    for i in range(iter_count-1):
                        if i%2==0:
                            if i!=0:
                                model.load_state_dict(model3.state_dict())
                                _output = model2(data)
                                loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)
                            else:
                                loss=loss_backprop

                            output1 = model(data)
                            loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)

                            model.backward_grad_correction_with_weight_change2_parametrized_relative_step_size_and_no_negations_in_step_direction(loss1, model2)
                        else:
                            model2.load_state_dict(model3.state_dict())

                            _output = model(data)
                            loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)

                            output1 = model2(data)
                            loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)

                            model2.backward_grad_correction_with_weight_change2_parametrized_relative_step_size_and_no_negations_in_step_direction(loss1, model)

                    if (iter_count)%2==1:
                        model.load_state_dict(model2.state_dict())

                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)

                    batch_counter += 1
                    if loss_backprop < loss2:
                        higher_loss_batch_counter += 1
                        # high_loss=True
                    elif loss_backprop > loss2:
                        lower_loss_batch_counter += 1
                    loss_improvement += float(loss_backprop.item() - loss2.item())

                    # if float(loss_backprop - _loss) != 0.:
                    #     relative_loss_improvement += float((loss_backprop - loss2) / abs(loss_backprop - _loss))
                    relative_loss_improvement_denominator += float(loss_backprop.item() - _loss.item())

                    total_loss_improvement += float(loss_backprop.item() - loss2.item())
                    total_relative_loss_improvement_denominator += float(loss_backprop.item() - _loss.item())
                else:
                    model2.load_state_dict(model.state_dict())

                    output = model2(data)
                    _loss = F.cross_entropy(output, target)#F.nll_loss(output, target)

                    opt2.zero_grad()

                    #_loss.backward()
                    if average_gradient_of_linear_layers_enhancement:
                        model2.backward(_loss)
                    else:
                        _loss.backward()

                    opt2.step()

                    if show_grad:
                        print(model2.layers[0].weight.grad[0][0][0])

                    _output = model2(data)
                    loss = F.cross_entropy(_output, target)#F.nll_loss(output, target)#todo check what this line changes
                    #model.load_state_dict(model2.state_dict())
                    #break

                    output1 = model(data)
                    loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)
                    #model.backward_grad_correction(loss1, model2, F.nll_loss, target)
                    model.backward_grad_correction_with_weight_change2_parametrized_relative_step_size_and_no_negations_in_step_direction(loss1, model2)

                    if show_grad:
                        print(model.layers[0].weight.grad[0][0][0])
                    if common_optimizer_parameters_enhancement:
                        #model.backward_grad_correction_with_weight_change2(loss1, model2, loss, F.cross_entropy, target)
                        model2.copy_grad_from(model)
                        opt2.step()
                    # model2.backward_grad_correction(loss,model)

                    #optimizer.step()
                    #opt2.zero_grad()

                    if False and additional_upgrade:
                        model2.load_state_dict(model.state_dict())
                        opt2.load_state_dict(optimizer.state_dict())

                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)#F.nll_loss(output2, target)

                    batch_counter+=1
                    if loss<loss2:
                        higher_loss_batch_counter+=1
                        #high_loss=True
                        if take_optimal_update:
                            model.load_state_dict(model2.state_dict())
                    elif loss>loss2:
                        lower_loss_batch_counter+=1
                    loss_improvement+=float(loss.item()-loss2.item())

                    relative_loss_improvement_denominator += float(loss.item() - _loss.item())

                    total_loss_improvement += float(loss.item()-loss2.item())
                    total_relative_loss_improvement_denominator += float(loss.item() - _loss.item())
            elif method==4:
                local_loss_diff_test=True

                avg_gradient_range_coef=0.01#10.

                if local_loss_diff_test:
                    local_diff_is_with_same_optimizer_parameters=False
                    model2.load_state_dict(model.state_dict())
                    if local_diff_is_with_same_optimizer_parameters:
                        opt2.load_state_dict(optimizer.state_dict())
                    opt2.zero_grad()
                    output2 = model2(data)
                    loss2_test = F.cross_entropy(output2, target)  # F.nll_loss(output, target)
                    loss2_test.backward()
                    opt2.step()
                    loss2_test = F.cross_entropy(model2(data), target)

                model2.load_state_dict(model.state_dict())
                #opt2.load_state_dict(optimizer.state_dict())


                _loss=None
                if local_loss_diff_test:
                    _loss = F.cross_entropy(model2(data), target)

                #with torch.no_grad()
                # model.translate_params(step_coefficient=-avg_gradient_range_coef,optimizer=optimizer)
                # out1=model.forward(data)
                # model.translate_params(step_coefficient=2.*avg_gradient_range_coef, optimizer=optimizer)
                # out2=model.forward2(data)
                # model.translate_params(step_coefficient=-avg_gradient_range_coef, optimizer=optimizer)
                translate_direction=(batch_idx%2)*2-1
                model.translate_params(step_coefficient=-avg_gradient_range_coef*translate_direction, optimizer=optimizer)
                out1 = model.forward(data)
                ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                model2.translate_params(step_coefficient=avg_gradient_range_coef*translate_direction, optimizer=optimizer)#opt2)
                out2 = model2.forward(data)
                #model.translate_params(step_coefficient=-avg_gradient_range_coef, optimizer=optimizer)

                loss_out1=F.cross_entropy(out1, target)
                loss_out2=F.cross_entropy(out2, target)
                model.backward_range_grad(loss_out1,loss_out2,model2)
                model.translate_params(step_coefficient=avg_gradient_range_coef*translate_direction, optimizer=optimizer)
                optimizer.step()

                # model2.load_state_dict(model.state_dict())
                # #opt2.load_state_dict(optimizer.state_dict())
                # # x=optimizer.state_dict()
                #
                # # opt2.param_groups[0]['params']=list(model2.parameters())
                # # load_adam_state(opt2,optimizer)
                # # x=optimizer.state_dict()
                # # opt2.zero_grad()
                # # optimizer.zero_grad()
                # # data.requires_grad=True
                #
                # output = model2(data)
                # _loss = F.cross_entropy(output, target)  # F.nll_loss(output, target)
                # # loss.requires_grad=True
                #
                # # model2.backward(loss)
                # # if i==0:
                # opt2.zero_grad()
                #
                # # _loss.backward()
                # if average_gradient_of_linear_layers_enhancement:
                #     model2.backward(_loss)
                # else:
                #     _loss.backward()
                #
                #     # model2.backward(_loss)
                #     # loss.backward(retain_graph=True)
                # # else:
                # #    model2.copy_grad_from(model)
                #
                # # loss.backward()
                # # model2.backward(loss)
                # opt2.step()
                #
                # if show_grad:
                #     print(model2.layers[0].weight.grad[0][0][0])
                #
                # _output = model2(data)
                # loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)#todo check what this line changes
                # # model.load_state_dict(model2.state_dict())
                # # break
                #
                # output1 = model(data)
                # loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)
                # # model.backward_grad_correction(loss1, model2, F.nll_loss, target)
                # model.backward_grad_correction_with_weight_change2(loss1, model2)
                #
                # if show_grad:
                #     print(model.layers[0].weight.grad[0][0][0])
                # if common_optimizer_parameters_enhancement:
                #     # model.backward_grad_correction_with_weight_change2(loss1, model2, loss, F.cross_entropy, target)
                #     model2.copy_grad_from(model)
                #     opt2.step()
                # # model2.backward_grad_correction(loss,model)
                #
                # # optimizer.step()
                # # opt2.zero_grad()
                #
                # if False and additional_upgrade:
                #     model2.load_state_dict(model.state_dict())
                #     opt2.load_state_dict(optimizer.state_dict())
                #
                # # model3.load_state_dict(model.state_dict().copy())
                # # opt3.load_state_dict(optimizer.state_dict())
                #
                # # additional_upgrade=False
                # # if additional_upgrade:
                # #     model3.load_state_dict(model.state_dict().copy())
                # #     opt3.load_state_dict(optimizer.state_dict())
                #
                # # optimizer.step()
                if local_loss_diff_test:
                    #_loss = F.cross_entropy(model2(data), target)
                    #loss2 = F.cross_entropy(model(data), target)

                    # opt2.zero_grad()
                    # output2 = model2(data)
                    # loss2_test = F.cross_entropy(output2, target)  # F.nll_loss(output, target)
                    # loss2_test.backward()
                    # opt2.step()
                    # loss2_test=F.cross_entropy(model2(data),target)


                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)

                    batch_counter += 1
                    if loss2_test < loss2:
                        higher_loss_batch_counter += 1
                        # high_loss=True
                        if take_optimal_update:
                            model.load_state_dict(model2.state_dict())
                    elif loss2_test > loss2:
                        lower_loss_batch_counter += 1
                    loss_improvement += float(loss2_test.item() - loss2.item())

                    relative_loss_improvement_denominator += float(loss2_test.item() - _loss.item())

                    total_loss_improvement += float(loss2_test.item() - loss2.item())
                    total_relative_loss_improvement_denominator += float(loss2_test.item() - _loss.item())
                loss=loss2

            elif method == 3:#!!!not ready yet
                model2.load_state_dict(model.state_dict())

                output = model2(data)
                _loss = F.cross_entropy(output, target)  # F.nll_loss(output, target)

                opt2.zero_grad()
                _loss.backward()

                opt2.step()

                _output = model2(data)
                loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)#todo check what this line changes

                output1 = model(data)
                loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)
                # model.backward_grad_correction(loss1, model2, F.nll_loss, target)
                model.backward_grad_correction_with_weight_change2(loss1, model2)

                if common_optimizer_parameters_enhancement:
                    model2.copy_grad_from(model)
                    opt2.step()

                output2 = model(data)
                loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)

                batch_counter += 1
                if loss < loss2:
                    higher_loss_batch_counter += 1
                    # high_loss=True
                elif loss > loss2:
                    lower_loss_batch_counter += 1
                loss_improvement += float(loss.item() - loss2.item())

                # if float(loss - _loss) != 0.:
                #     relative_loss_improvement += float((loss - loss2) / abs(loss - _loss))
                relative_loss_improvement_denominator+=float(loss.item()-_loss.item())

                total_loss_improvement+=loss.item() - loss2.item()
                total_relative_loss_improvement_denominator+=loss.item()-_loss.item()



            elif method == 2:
                global with_optimizer_parameter_copy

                model.gradient_factor_simple_layers = model.gradient_factor
                high_loss = False
                iter_count=1
                for i in range(iter_count):
                    model2.load_state_dict(model.state_dict())
                    # x=optimizer.state_dict()
                    if with_optimizer_parameter_copy:
                        opt2.load_state_dict(optimizer.state_dict())
                    # opt2.param_groups[0]['params']=list(model2.parameters())
                    # load_adam_state(opt2,optimizer)
                    # x=optimizer.state_dict()
                    # opt2.zero_grad()
                    # optimizer.zero_grad()
                    # data.requires_grad=True

                    output = model2(data)
                    loss = F.cross_entropy(output, target)  # F.nll_loss(output, target)
                    # loss.requires_grad=True

                    # model2.backward(loss)
                    if i == 0:
                        opt2.zero_grad()
                        loss.backward(retain_graph=True)
                    else:
                        model2.copy_grad_from(model)

                    opt2.step()

                    output1 = model(data)
                    loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)

                    output = model2(data)
                    loss = F.cross_entropy(output, target)  # F.nll_loss(output, target)#todo check what this line changes

                    #model.backward_grad_correction(loss1, model2, F.cross_entropy, target)
                    model.backward_grad_correction_with_weight_change2(loss1,model2,weight_change=False,accumulate_gradients=False,gradient_only_modification=True)

                    if False and additional_upgrade:
                        model2.load_state_dict(model.state_dict())
                        opt2.load_state_dict(optimizer.state_dict())

                    additional_upgrade = False
                    if additional_upgrade:
                        model3.load_state_dict(model.state_dict().copy())
                        opt3.load_state_dict(optimizer.state_dict())

                    optimizer.step()
                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)
                    #
                    # batch_counter += 1
                    # if loss < loss2:
                    #     higher_loss_batch_counter += 1
                    #     high_loss = True
                    # elif loss > loss2:
                    #     lower_loss_batch_counter += 1

                    batch_counter += 1
                    if loss < loss2:
                        higher_loss_batch_counter += 1
                        high_loss=True
                    elif loss > loss2:
                        lower_loss_batch_counter += 1
                    loss_improvement += float(loss.item() - loss2.item())

                    # if float(loss - loss1) != 0.:
                    #     relative_loss_improvement += float((loss - loss2) / abs(loss - loss1))
                    relative_loss_improvement_denominator += float(loss.item() - loss1.item())

                    total_loss_improvement += float(loss.item() - loss2.item())
                    total_relative_loss_improvement_denominator += float(loss.item() - loss1.item())

                    if additional_upgrade and loss1 < loss2:
                        model.load_state_dict(model3.state_dict())
                        optimizer.load_state_dict(opt3.state_dict())
                        break

                model.loss_signal(high_loss)
                model2.gradient_factor = model.gradient_factor
                model2.gradient_factor_simple_layers = model.gradient_factor_simple_layers
                model2.step_factor = model.step_factor
            elif method==1:
                # def load_adam_state(opt2:optim.Adam,opt:optim.Adam):
                #     opt_values=list(opt.state.values())
                #     opt2_values=list(opt2.state.values())
                #     #for key,value in opt2.state.items():
                #     soft_copy_factor=0.01
                #     for i in range(len(opt2_values)):
                #         opt2_values[i]['exp_avg']=opt2_values[i]['exp_avg']*(1-soft_copy_factor)+soft_copy_factor*opt_values[i]['exp_avg'].clone().detach()
                #         opt2_values[i]['exp_avg_sq'] = opt2_values[i]['exp_avg_sq']*(1-soft_copy_factor)+soft_copy_factor*opt_values[i]['exp_avg_sq'].clone().detach()
                #model.step_factor=1.

                if iter_count>=3:
                    model2.load_state_dict(model.state_dict())
                    model3.load_state_dict(model.state_dict())

                    output = model2(data)
                    _loss = F.cross_entropy(output, target)  # F.nll_loss(output, target)

                    opt2.zero_grad()
                    _loss.backward()
                    # if i == 0:
                    #     opt2.zero_grad()
                    #     _loss.backward()
                    #     # model2.backward(_loss)
                    #     # loss.backward(retain_graph=True)
                    # else:
                    #     model2.copy_grad_from(model)

                    opt2.step()
                    _output = model2(data)
                    loss_backprop=F.cross_entropy(_output, target)

                    for i in range(iter_count-1):
                        if i%2==0:
                            if i!=0:
                                model.load_state_dict(model3.state_dict())
                                _output = model2(data)
                                loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)
                            else:
                                loss=loss_backprop

                            output1 = model(data)
                            loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)

                            model.backward_grad_correction_with_weight_change2(loss1, model2)
                        else:
                            model2.load_state_dict(model3.state_dict())

                            _output = model(data)
                            loss = F.cross_entropy(_output, target)  # F.nll_loss(output, target)

                            output1 = model2(data)
                            loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)

                            model2.backward_grad_correction_with_weight_change2(loss1, model)

                    if (iter_count)%2==1:
                        model.load_state_dict(model2.state_dict())

                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)  # F.nll_loss(output2, target)

                    batch_counter += 1
                    if loss_backprop < loss2:
                        higher_loss_batch_counter += 1
                        # high_loss=True
                    elif loss_backprop > loss2:
                        lower_loss_batch_counter += 1
                    loss_improvement += float(loss_backprop.item() - loss2.item())

                    # if float(loss_backprop - _loss) != 0.:
                    #     relative_loss_improvement += float((loss_backprop - loss2) / abs(loss_backprop - _loss))
                    relative_loss_improvement_denominator += float(loss_backprop.item() - _loss.item())

                    total_loss_improvement += float(loss_backprop.item() - loss2.item())
                    total_relative_loss_improvement_denominator += float(loss_backprop.item() - _loss.item())
                else:
                    #model.gradient_factor_simple_layers=model.gradient_factor
                    #high_loss=False
                    #optimizer.zero_grad()
                    #for i in range(iter_count):
                    model2.load_state_dict(model.state_dict())
                    #opt2.load_state_dict(optimizer.state_dict())
                    #x=optimizer.state_dict()

                    #opt2.param_groups[0]['params']=list(model2.parameters())
                    #load_adam_state(opt2,optimizer)
                    #x=optimizer.state_dict()
                    #opt2.zero_grad()
                    #optimizer.zero_grad()
                    # data.requires_grad=True

                    output = model2(data)
                    _loss = F.cross_entropy(output, target)#F.nll_loss(output, target)
                    # loss.requires_grad=True

                    # model2.backward(loss)
                    #if i==0:
                    opt2.zero_grad()

                    #_loss.backward()
                    if average_gradient_of_linear_layers_enhancement:
                        model2.backward(_loss)
                    else:
                        _loss.backward()

                        #model2.backward(_loss)
                        #loss.backward(retain_graph=True)
                    #else:
                    #    model2.copy_grad_from(model)

                    # loss.backward()
                    # model2.backward(loss)
                    opt2.step()

                    if show_grad:
                        print(model2.layers[0].weight.grad[0][0][0])

                    _output = model2(data)
                    loss = F.cross_entropy(_output, target)#F.nll_loss(output, target)#todo check what this line changes
                    #model.load_state_dict(model2.state_dict())
                    #break

                    output1 = model(data)
                    loss1 = F.cross_entropy(output1, target)  # F.nll_loss(output1, target)
                    #model.backward_grad_correction(loss1, model2, F.nll_loss, target)
                    model.backward_grad_correction_with_weight_change2(loss1, model2)

                    if show_grad:
                        print(model.layers[0].weight.grad[0][0][0])
                    if common_optimizer_parameters_enhancement:
                        #model.backward_grad_correction_with_weight_change2(loss1, model2, loss, F.cross_entropy, target)
                        model2.copy_grad_from(model)
                        opt2.step()
                    # model2.backward_grad_correction(loss,model)

                    #optimizer.step()
                    #opt2.zero_grad()

                    if False and additional_upgrade:
                        model2.load_state_dict(model.state_dict())
                        opt2.load_state_dict(optimizer.state_dict())

                    # model3.load_state_dict(model.state_dict().copy())
                    # opt3.load_state_dict(optimizer.state_dict())

                    # additional_upgrade=False
                    # if additional_upgrade:
                    #     model3.load_state_dict(model.state_dict().copy())
                    #     opt3.load_state_dict(optimizer.state_dict())

                    #optimizer.step()
                    output2 = model(data)
                    loss2 = F.cross_entropy(output2, target)#F.nll_loss(output2, target)

                    batch_counter+=1
                    if loss<loss2:
                        higher_loss_batch_counter+=1
                        #high_loss=True
                        if take_optimal_update:
                            model.load_state_dict(model2.state_dict())
                    elif loss>loss2:
                        lower_loss_batch_counter+=1
                    loss_improvement+=float(loss.item()-loss2.item())

                    relative_loss_improvement_denominator += float(loss.item() - _loss.item())

                    total_loss_improvement += float(loss.item()-loss2.item())
                    total_relative_loss_improvement_denominator += float(loss.item() - _loss.item())
                    # if float(loss-_loss)!=0.:
                    #     relative_loss_improvement+=float((loss-loss2)/abs(loss-_loss))
                    #
                    #     #opt2.load_state_dict(optimizer.state_dict())
                    #
                    #     additional_upgrade = False
                    #     if additional_upgrade and loss < loss2:
                    #         #print('aaaaaaaaaaaaaaa')
                    #         model.load_state_dict(model2.state_dict())
                    #         #optimizer.load_state_dict(opt2.state_dict())
                    #         # output = model(data)
                    #         # loss = F.nll_loss(output, target)
                    #         # loss.backward(retain_graph=True)
                    #         # optimizer.step()
                    #         #loss=loss2
                    #         #todo: test with no weight change
                    #         #break
                    #     #loss = loss2


                # model.loss_signal(high_loss)
                # model2.gradient_factor=model.gradient_factor
                # model2.gradient_factor_simple_layers=model.gradient_factor_simple_layers
                # model2.step_factor=model.step_factor
                #model.load_state_dict(model2.state_dict())


                # model2.load_state_dict(model.state_dict())
                # opt2.load_state_dict(optimizer.state_dict())
                # opt2.step()
                # model.load_state_dict(model2.state_dict())
                # optimizer.load_state_dict(opt2.state_dict())


                #optimizer.step()
            else:
                optimizer.zero_grad()
                # data.requires_grad=True
                output = model(data)
                loss = F.cross_entropy(output, target)#F.nll_loss(output, target)
                # loss.requires_grad=True
                #print(hash(model.named_parameters()))
                #print(dict(model.named_parameters()).keys())

                #torch.manual_seed(1)
                loss.backward()
                #model.backward(loss)
                #torch.manual_seed(1)
                #x=model.state_dict()
                #print(x)

                optimizer.step()
                #print(dict(model.named_parameters()))
                if show_grad:
                    print(model.layers[0].weight.grad[0][0][0])

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() if hasattr(loss,'item') else loss))
                if batch_counter!=0:
                    print("Higher loss batches: "+str(higher_loss_batch_counter)+"/"+str(batch_counter)+"="+str(higher_loss_batch_counter/batch_counter)+"    Lower loss batch ratio: "+str(lower_loss_batch_counter/batch_counter)+" Avg loss improvement: "+str(loss_improvement/batch_counter)+" Avg relative loss improvement: "+str(loss_improvement/abs(relative_loss_improvement_denominator))+" Total avg relative loss improvement: "+str(total_loss_improvement/abs(total_relative_loss_improvement_denominator)))
        if batch_counter!=0:
            if model.stats is None:
                model.stats={}

            if step_length_stats:
                key = 'relative_step_length_l1'
                if key not in model.stats:
                    model.stats[key] = {}
                if epoch not in model.stats[key]:
                    model.stats[key][epoch] = []
                model.stats[key][epoch].append(step_length_l1/step_length_base_l1 if step_length_base_l1!=0. else np.inf)
                key = 'relative_step_length_l2'
                if key not in model.stats:
                    model.stats[key] = {}
                if epoch not in model.stats[key]:
                    model.stats[key][epoch] = []
                model.stats[key][epoch].append(step_length_l2/step_length_base_l2 if step_length_base_l2!=0. else np.inf)

            key='train_higher_loss_batch_ratio'
            if key not in model.stats:
                model.stats[key]={}
            if epoch not in model.stats[key]:
                model.stats[key][epoch]=[]
            model.stats[key][epoch].append(higher_loss_batch_counter/batch_counter)

            key = 'train_lower_loss_batch_ratio'
            if key not in model.stats:
                model.stats[key] = {}
            if epoch not in model.stats[key]:
                model.stats[key][epoch] = []
            model.stats[key][epoch].append(lower_loss_batch_counter / batch_counter)

            key = 'train_same_loss_batch_ratio'
            if key not in model.stats:
                model.stats[key] = {}
            if epoch not in model.stats[key]:
                model.stats[key][epoch] = []
            model.stats[key][epoch].append((batch_counter-lower_loss_batch_counter-higher_loss_batch_counter) / batch_counter)

            key = 'train_batch_avg_loss_improvement'
            if key not in model.stats:
                model.stats[key] = {}
            if epoch not in model.stats[key]:
                model.stats[key][epoch] = []
            model.stats[key][epoch].append(
                loss_improvement / batch_counter)

            key = 'train_batch_avg_relative_loss_improvement'
            if key not in model.stats:
                model.stats[key] = {}
            if epoch not in model.stats[key]:
                model.stats[key][epoch] = []
            model.stats[key][epoch].append(
                loss_improvement/abs(relative_loss_improvement_denominator))

            key = 'train_avg_relative_loss_improvement'
            if key not in model.stats:
                model.stats[key] = {}
            if epoch not in model.stats[key]:
                model.stats[key][epoch] = []
            model.stats[key][epoch].append(
                total_loss_improvement / abs(total_relative_loss_improvement_denominator))

    def test(model, device, test_loader,train_loader,val_loader=None,epoch=1):
        if model.stats is None:
            model.stats={}

        if 'train_accuracy' not in model.stats:
            model.stats['train_accuracy']={}
        if calculate_test_stats and 'test_accuracy' not in model.stats:
            model.stats['test_accuracy'] = {}
        if val_loader is not None and 'val_accuracy' not in model.stats:
            model.stats['val_accuracy'] = {}

        if 'train_loss' not in model.stats:
            model.stats['train_loss'] = {}
        if calculate_test_stats and 'test_loss' not in model.stats:
            model.stats['test_loss'] = {}
        if val_loader is not None and 'val_loss' not in model.stats:
            model.stats['val_loss'] = {}

        if epoch  not in model.stats['train_accuracy']:
            model.stats['train_accuracy'][epoch]=[]
        if epoch  not in model.stats['train_loss']:
            model.stats['train_loss'][epoch]=[]
        if calculate_test_stats:
            if epoch  not in model.stats['test_accuracy']:
                model.stats['test_accuracy'][epoch]=[]
            if epoch  not in model.stats['test_loss']:
                model.stats['test_loss'][epoch]=[]
        if val_loader is not None:
            if epoch  not in model.stats['val_accuracy']:
                model.stats['val_accuracy'][epoch]=[]
            if epoch  not in model.stats['val_loss']:
                model.stats['val_loss'][epoch]=[]

        mapping = None  # class mapping specifically for mode1 nr 13
        logit_mask = None  # logit mask specifically for model nr 13
        if model_nr == 13 or model_nr == 14 or model_nr == 15 or model_nr==16 or model_nr==17 or model_nr==18:
            mapping = get_approx_optimal_class_mapping(train_loader)
            mapping = mapping.long().to(device)
            logit_mask = get_logit_mask(mapping)
            logit_mask = logit_mask.to(device)

        model.eval()

        if calculate_test_stats:
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    if mapping is not None:
                        target=mapping[target]
                    output = model(data)
                    if logit_mask is not None:
                        output+=logit_mask
                    #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print('\nTest set: Avg loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
            model.stats['test_accuracy'][epoch].append(100. * correct / len(test_loader.dataset))
            model.stats['test_loss'][epoch].append(test_loss)

        if calculate_train_stats:
            train_loss = 0
            correct = 0
            length = 0
            with torch.no_grad():
                for data, target in train_loader:
                    length += data.shape[0]
                    data, target = data.to(device), target.to(device)
                    if mapping is not None:
                        target=mapping[target]
                    output = model(data)
                    if logit_mask is not None:
                        output+=logit_mask
                    #train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    train_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            train_loss /= length


            print('\nTraining set: Avg loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
                train_loss, correct, length,
                100. * correct / length))
            model.stats['train_accuracy'][epoch].append(100. * correct / length)
            model.stats['train_loss'][epoch].append(train_loss)

            if val_loader is not None:
                val_loss = 0
                correct = 0
                length=0
                with torch.no_grad():
                    for data, target in val_loader:
                        length+=data.shape[0]
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        # train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                        val_loss += F.cross_entropy(output, target, reduction='sum').item()
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()

                val_loss /= length

                print('\nValidation set: Avg loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
                    val_loss, correct, length,
                    100. * correct / length))
                model.stats['val_accuracy'][epoch].append(100. * correct / length)
                model.stats['val_loss'][epoch].append(val_loss)



    def evaluate_stats(stats):
        # # return -stats['train_loss_avg']
        if ('train_loss_onthefly_aggregated' not in stats or len(stats['train_loss_onthefly_aggregated'].items())==0) and ('train_loss' not in stats or len(stats['train_loss'].items())==0):
            return np.Inf

        # if not calculate_train_stats:
        #     # return np.Inf
        #
        #     min_loss = np.Inf  # minimal training loss in any epoch
        #     min_epoch=0
        #     for epoch_num, score in stats['train_loss_onthefly_aggregated'].items():
        #         if min_loss > score:
        #             min_loss = score
        #             min_epoch=epoch_num
        #         # if min_loss>sum(scores)/len(scores):
        #         #     min_loss=sum(scores)/len(scores)
        #     if min_loss==0.:
        #         return 1./min_epoch#the lowest is the first epoch with 0 loss, the higher is the evaluation
        #     return -min_loss  # "-" to minimize min_loss value instead of default maximization
        #
        # if cross_val:
        #     min_loss=np.Inf #minimal training loss in any epoch
        #     for epoch_num,score in stats['val_loss_aggregated'].items():
        #         if min_loss>score:
        #             min_loss=score
        #         # if min_loss>sum(scores)/len(scores):
        #         #     min_loss=sum(scores)/len(scores)
        #     return -min_loss #"-" to minimize min_loss value instead of default maximization
        # else:
        #     min_loss = np.Inf  # minimal training loss in any epoch
        #     for epoch_num, score in stats['train_loss_aggregated'].items():
        #         if min_loss > score:
        #             min_loss = score
        #         # if min_loss>sum(scores)/len(scores):
        #         #     min_loss=sum(scores)/len(scores)
        #     return -min_loss  # "-" to minimize min_loss value instead of default maximization
        if not calculate_train_stats:
            return -sum(stats['train_loss_onthefly_min'])/len(stats['train_loss_onthefly_min'])

        if cross_val:
            return -sum(stats['val_loss_min'])/len(stats['val_loss_min'])
        else:
            return -sum(stats['train_loss_min'])/len(stats['train_loss_min'])

    def stop_criteria(stats):
        #return evaluate_stats(stats)<-3.
        #return evaluate_stats(stats)<-5.
        return False
    def process_stats(model):
        stats={}
        for key,value in model.stats.items():
            if key=='params':
                stats[key]=copy.deepcopy(model.stats[key])
                continue
            stats[key+"_aggregated"]={}
            stats[key] = {}
            all_scores=[]
            for epoch_num,list_of_scores in value.items():
                if len(list_of_scores)!=0:
                    stats[key+"_aggregated"][epoch_num]=sum(list_of_scores)/len(list_of_scores)
                    all_scores+=list_of_scores
                    stats[key][epoch_num]=copy.deepcopy(list_of_scores)
            if len(all_scores)!=0:
                stats[str(key)+'_avg']=sum(all_scores)/len(all_scores)

            #epochs=list(value.keys())
            lists_of_scores=list(value.values())
            stats[key + "_max"]=[]
            stats[key + "_min"]=[]
            for training_num in range(len(lists_of_scores[0])):
                val_max=-np.Inf
                val_min=np.Inf
                for epoch in range(len(lists_of_scores)):
                    if len(lists_of_scores[epoch])==training_num:
                        continue
                    val=lists_of_scores[epoch][training_num]
                    if val_min>val:
                        val_min=val
                    if val_max<val:
                        val_max=val
                stats[key + "_max"].append(val_max)
                stats[key + "_min"].append(val_min)
            if len(stats[key + "_max"])!=0:
                stats[key + "_max_avg"]=sum(stats[key + "_max"])/len(stats[key + "_max"])
                stats[key + "_min_avg"] = sum(stats[key + "_min"]) / len(stats[key + "_min"])


        stats['gradient_factor']=model.gradient_factor
        stats['step_factor'] = model.step_factor

        if cross_val:
            lists_of_scores = list(model.stats['val_loss'].values())
            for training_num in range(len(lists_of_scores[0])):
                val_max = -np.Inf
                val_min = np.Inf
                epoch_min=-1
                for epoch in range(len(lists_of_scores)):
                    if len(lists_of_scores[epoch]) == training_num:
                        continue
                    val = lists_of_scores[epoch][training_num]
                    if val_min > val:
                        val_min = val
                        epoch_min=epoch+1
                    if val_max < val:
                        val_max = val
                if calculate_test_stats:
                    stats['test_loss_validation_optimal']=model.stats['test_loss'][epoch_min]
                if 'validation_optimal_epoch' not in stats:
                    stats['validation_optimal_epoch']=[]
                stats['validation_optimal_epoch'].append(epoch_min)
                #stats[key + "_max"].append(val_max)
                #stats[key + "_min"].append(val_min)
            stats['validation_optimal_epoch_aggregated']=sum(stats['validation_optimal_epoch'])/len(stats['validation_optimal_epoch'])

        return stats
    def write_to_file(name,text,mode='a'):
        with open(name, mode) as myfile:
            myfile.write(text+'\n')

    def save_model(model,name):
        if model_nr>=13:
            import sys
            sys.setrecursionlimit(3000)
        torch.save(model,name)

    def load_model(name):
        return torch.load(name)

    def rand_between(a,b):
        return torch.FloatTensor(1).uniform_(a, b)[0]

    hyperparameter_index=-1

    # class State:
    #     hyperparameter_index=-1
    #     actual_stats={}
    #     trainings_different_model=0
    #     trainings_same_model=0
    def save_state(file_name,data):
        #write_to_file(file_name,data)
        torch.set_printoptions(profile="full")#otherwise "default"
        tmp_name='tmp_'+file_name
        with open(tmp_name, 'w') as myfile:
            myfile.write(str(data).replace("tensor",'torch.tensor'))
            myfile.flush()
            os.fsync(myfile.fileno())
        torch.set_printoptions(profile="default")
        os.replace(tmp_name,file_name)#atomic operation
        # if os.path.isfile(tmp_name):
        #     os.remove(tmp_name)
        #     #print("Removing tmp file")
    def read_state(file_name):
        try:
            with open(file_name, 'r') as f:
                s = f.read()
                #return ast.literal_eval(s)
                return eval(s)
        except Exception as e:  # works on python 3.x
            print(repr(e))
        return None

    def main():
        #device = torch.device("cpu")
        global method
        global batch_size
        lr=1#0.007 #todo: change
        gamma=0.7 #todo: set optimal gamma
        train_kwargs = {'batch_size': batch_size}
        test_kwargs = {'batch_size': batch_size}

        # seed=23
        torch.manual_seed(seed)

        # epochs=2
        # if model_nr == 3:
        #     epochs=2
        # elif model_nr == 4:
        #     epochs=2
        # elif model_nr == 5:
        #     epochs=30
        # elif model_nr == 6:
        #     epochs=40

        #global trainings_same_model
        #global trainings_different_model
        train_loader:torch.utils.data.DataLoader=None
        test_loader:torch.utils.data.DataLoader=None

        convert_to_datatype=lambda x: x.to(d_type)
        if dataset=='mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,))
                convert_to_datatype
            ])
            dataset1 = datasets.MNIST('../data', train=True, download=True,
                               transform=transform)
            dataset2 = datasets.MNIST('../data', train=False,
                               transform=transform)
            train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, **train_kwargs)
            test_loader = torch.utils.data.DataLoader(dataset2, shuffle=True, **test_kwargs)
        elif dataset=='fashion_mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,))
                convert_to_datatype
            ])
            dataset1 = datasets.FashionMNIST('../data', train=True, download=True,
                                      transform=transform)
            dataset2 = datasets.FashionMNIST('../data', train=False,
                                      transform=transform)
            train_loader = torch.utils.data.DataLoader(dataset1, shuffle=True, **train_kwargs)
            test_loader = torch.utils.data.DataLoader(dataset2, shuffle=True, **test_kwargs)
        elif dataset=='imdb':
            train_loader, test_loader = imdb_utils.get_preprocessed_IMDB(d_type=d_type)
            train_loader=torch.utils.data.DataLoader(train_loader.dataset, shuffle=True, **train_kwargs)
            test_loader=torch.utils.data.DataLoader(test_loader.dataset, shuffle=True, **test_kwargs)
        elif dataset=='imagenet_ood':
            train_loader,test_loader=get_dataloaders(batch_size=batch_size)

        def get_layers():
            if model_nr==3:
                return [nn.Conv2d(1, 16, 3, 1,dtype=d_type),
                        nn.ReLU(),
                        #nn.BatchNorm2d(16),
                        nn.Conv2d(16, 16, 3, 1,dtype=d_type),
                        nn.ReLU(),
                        #nn.BatchNorm2d(16),
                        nn.MaxPool2d(2),
                        # nn.Dropout(0.25),
                        nn.Flatten(),
                        nn.Linear(2304, 32,dtype=d_type),
                        nn.ReLU(),
                        #nn.BatchNorm1d(32),
                        # nn.Dropout(0.5),
                        nn.Linear(32, 10,dtype=d_type),
                        #nn.Softmax(dim=1)#nn.LogSoftmax(dim=1)
                        ]
            elif model_nr==4:
                return [nn.Conv2d(1, 8, 3, 1,dtype=d_type),
                        nn.ReLU(),
                        #nn.BatchNorm2d(8),
                        # nn.MaxPool2d(2),
                        nn.Conv2d(8, 8, 3, 1,dtype=d_type),
                        nn.ReLU(),
                        #nn.BatchNorm2d(8),
                        nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2,dtype=d_type),
                        nn.ReLU(),
                        #nn.BatchNorm2d(16),

                        nn.Conv2d(16, 16, 3, 1,dtype=d_type),
                        nn.ReLU(),
                        #nn.BatchNorm2d(16),
                        # nn.MaxPool2d(2),
                        nn.Conv2d(16, 16, 3, 1,dtype=d_type),
                        nn.ReLU(),
                        #nn.BatchNorm2d(16),
                        nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2,dtype=d_type),
                        nn.ReLU(),
                        #nn.BatchNorm2d(16),

                        # nn.MaxPool2d(2),
                        # nn.Dropout(0.25),
                        nn.Flatten(),
                        nn.Linear(256, 10,dtype=d_type),
                        # nn.ReLU(),
                        # nn.Dropout(0.5),
                        # nn.Linear(32, 10),
                        #nn.Softmax(dim=1)
                        ]
            elif model_nr==5:
                return [nn.Conv2d(1, 8, 3, 1,dtype=d_type),
                        nn.ELU(),
                        # nn.BatchNorm2d(16),
                        nn.MaxPool2d(2),
                        nn.Conv2d(8, 16, 3, 1,dtype=d_type),
                        nn.ELU(),
                        # nn.BatchNorm2d(16),
                        nn.MaxPool2d(2),
                        # nn.Dropout(0.25),
                        nn.Flatten(),
                        nn.Linear(400, 32,dtype=d_type),  # nn.Linear(2304, 32),
                        nn.ELU(),
                        # nn.BatchNorm1d(32),
                        # nn.Dropout(0.5),
                        nn.Linear(32, 10,dtype=d_type),
                        # nn.Softmax(dim=1)#nn.LogSoftmax(dim=1)
                        ]
                # return [nn.Conv2d(1, 8, 3, 1),
                #         nn.Hardswish(),
                #         #nn.BatchNorm2d(16),
                #         nn.MaxPool2d(2),
                #         nn.Conv2d(8, 16, 3, 1),
                #         nn.Hardswish(),
                #         #nn.BatchNorm2d(16),
                #         nn.MaxPool2d(2),
                #         # nn.Dropout(0.25),
                #         nn.Flatten(),
                #         nn.Linear(400, 32),#nn.Linear(2304, 32),
                #         nn.Hardswish(),
                #         #nn.BatchNorm1d(32),
                #         # nn.Dropout(0.5),
                #         nn.Linear(32, 10),
                #         #nn.Softmax(dim=1)#nn.LogSoftmax(dim=1)
                #         ]
            elif model_nr==6:
                # return [nn.Conv2d(1, 8, 3, 1),
                #         nn.ELU(),
                #         nn.BatchNorm2d(8),
                #         # nn.MaxPool2d(2),
                #         nn.Conv2d(8, 8, 3, 1),
                #         nn.ELU(),
                #         nn.BatchNorm2d(8),
                #         nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
                #         nn.ELU(),
                #         nn.BatchNorm2d(16),
                #
                #         nn.Conv2d(16, 16, 3, 1),
                #         nn.ELU(),
                #         nn.BatchNorm2d(16),
                #         # nn.MaxPool2d(2),
                #         nn.Conv2d(16, 16, 3, 1),
                #         nn.ELU(),
                #         nn.BatchNorm2d(16),
                #         nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),
                #         nn.ELU(),
                #         nn.BatchNorm2d(16),
                #
                #         # nn.MaxPool2d(2),
                #         # nn.Dropout(0.25),
                #         nn.Flatten(),
                #         nn.Linear(256, 10),
                #         # nn.ReLU(),
                #         # nn.Dropout(0.5),
                #         # nn.Linear(32, 10),
                #         nn.Softmax(dim=1)]
                return [nn.Conv2d(1, 8, 3, 1,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(8),
                        # nn.MaxPool2d(2),
                        nn.Conv2d(8, 8, 3, 1,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(8),
                        nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(16),

                        nn.Conv2d(16, 16, 3, 1,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(16),
                        # nn.MaxPool2d(2),
                        nn.Conv2d(16, 16, 3, 1,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(16),
                        nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(16),

                        # nn.MaxPool2d(2),
                        # nn.Dropout(0.25),
                        nn.Flatten(),
                        nn.Linear(256, 10,dtype=d_type),
                        # nn.ReLU(),
                        # nn.Dropout(0.5),
                        # nn.Linear(32, 10),
                        #nn.Softmax(dim=1)
                        ]
            elif model_nr==7:
                return [nn.Conv2d(1, 8, 3, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1,1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.Hardswish(),


                        ##nn.BatchNorm2d(8),
                        # nn.MaxPool2d(2),
                        nn.Conv2d(8, 8, 3, 1,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(8),
                        nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(16),

                        nn.Conv2d(16, 16, 3, 1,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(16),
                        # nn.MaxPool2d(2),
                        nn.Conv2d(16, 16, 3, 1,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(16),
                        nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(16),

                        # nn.MaxPool2d(2),
                        # nn.Dropout(0.25),
                        nn.Flatten(),
                        nn.Linear(256, 10,dtype=d_type),
                        # nn.ReLU(),
                        # nn.Dropout(0.5),
                        # nn.Linear(32, 10),
                        #nn.Softmax(dim=1)
                        ]
            elif model_nr==8:
                return [nn.Conv2d(1, 8, 3, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1,1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),
                        nn.Conv2d(8, 8, 3, 1, 1,dtype=d_type),
                        nn.ELU(),


                        ##nn.BatchNorm2d(8),
                        # nn.MaxPool2d(2),
                        nn.Conv2d(8, 8, 3, 1,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(8),
                        nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(16),

                        nn.Conv2d(16, 16, 3, 1,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(16),
                        # nn.MaxPool2d(2),
                        nn.Conv2d(16, 16, 3, 1,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(16),
                        nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2,dtype=d_type),
                        nn.ELU(),
                        #nn.BatchNorm2d(16),

                        # nn.MaxPool2d(2),
                        # nn.Dropout(0.25),
                        nn.Flatten(),
                        nn.Linear(256, 10,dtype=d_type),
                        # nn.ReLU(),
                        # nn.Dropout(0.5),
                        # nn.Linear(32, 10),
                        #nn.Softmax(dim=1)
                        ]
            elif model_nr==9:
                layers= [
                    nn.Conv2d(1, 8, 3, 1,dtype=d_type),
                    nn.ELU(),
                    # nn.BatchNorm2d(16),
                    nn.MaxPool2d(2),
                    nn.Conv2d(8, 16, 3, 1,dtype=d_type),
                    nn.ELU(),
                    # nn.BatchNorm2d(16),
                    #nn.BatchNorm2d(num_features=16),
                    nn.MaxPool2d(2),
                    # nn.Dropout(0.25),
                    nn.Flatten(),
                    nn.Linear(400, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),
                    nn.Tanh(),
                    nn.Linear(10, 10,dtype=d_type),



                    #nn.Conv2d(1, 8, 3, 1),
                #         nn.ELU(),
                #         # nn.BatchNorm2d(16),
                #         nn.MaxPool2d(2),
                #         nn.Conv2d(8, 16, 3, 1),
                #         nn.ELU(),
                #         # nn.BatchNorm2d(16),
                #         nn.MaxPool2d(2),
                #         # nn.Dropout(0.25),
                #         nn.Flatten(),
                #         nn.Linear(400, 32),  # nn.Linear(2304, 32),
                #         nn.ELU(),
                #         # nn.BatchNorm1d(32),
                #         # nn.Dropout(0.5),
                #         nn.Linear(32, 10),
                #         # nn.Softmax(dim=1)#nn.LogSoftmax(dim=1)
                        ]
                init_weights(layers, nn.init.xavier_uniform_)
                return layers
            elif model_nr == 10:
                slope=0.1
                layers=[nn.Conv2d(1, 8, 3, 1,dtype=d_type),
                        nn.LeakyReLU(slope),
                        # nn.BatchNorm2d(8),
                        # nn.MaxPool2d(2),
                        nn.Conv2d(8, 8, 3, 1,dtype=d_type),
                        nn.LeakyReLU(slope),
                        # nn.BatchNorm2d(8),
                        nn.Conv2d(8, 8, kernel_size=5, stride=2, padding=2,dtype=d_type),
                        nn.LeakyReLU(slope),
                        # nn.BatchNorm2d(16),

                        nn.Conv2d(8, 8, 3, 1,dtype=d_type),
                        nn.LeakyReLU(slope),
                        # nn.BatchNorm2d(16),
                        # nn.MaxPool2d(2),
                        ]
                for _ in range(32):
                    layers+=(lambda:[
                           nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1,dtype=d_type),
                           nn.LeakyReLU(slope), ])()
                layers+=[

                           nn.Conv2d(8, 8, 3, 1,dtype=d_type),
                           nn.LeakyReLU(slope),
                           # nn.BatchNorm2d(16),
                           nn.Conv2d(8, 8, kernel_size=5, stride=2, padding=2,dtype=d_type),
                           nn.LeakyReLU(slope),
                           # nn.BatchNorm2d(16),

                           # nn.MaxPool2d(2),
                           # nn.Dropout(0.25),
                           nn.Flatten(),
                           nn.Linear(128, 30,dtype=d_type),
                           nn.Sigmoid(),
                           nn.Linear(30, 10,dtype=d_type),
                           # nn.ReLU(),
                           # nn.Dropout(0.5),
                           # nn.Linear(32, 10),
                           # nn.Softmax(dim=1)
                       ]
                return layers
            elif model_nr == 11:
                layers=imdb_utils.get_model_layers(d_type=d_type,model_nr=1)
                init_weights(layers,nn.init.xavier_uniform_)
                init_weights(layers,nn.init.xavier_uniform_,avg_gain=1.66666667)
                return layers
            elif model_nr == 12:
                layers=imdb_utils.get_model_layers(d_type=d_type,model_nr=2)
                init_weights(layers,nn.init.xavier_uniform_,avg_gain=1.66666667)#2.85)
                return layers
            elif model_nr == 13:
                resnet152=load_resnet()
                module_graph=convert_resnet_to_module_graph(resnet152)
                return module_graph
            elif model_nr == 14:
                #resnet_152_gelu=load_resnet_with_GELU()
                resnet152=load_resnet()
                module_graph = convert_resnet_to_module_graph(resnet152)
                module_graph_GELU=switch_RELU_to_GELU_in_dependency_graph(module_graph)
                return module_graph_GELU
            elif model_nr == 15:
                #resnet_152_gelu=load_resnet_with_GELU()
                resnet152=load_resnet()
                module_graph = convert_resnet_to_module_graph(resnet152)
                module_graph_SiLU=switch_RELU_to_SiLU_in_dependency_graph(module_graph)
                return module_graph_SiLU
            elif model_nr == 16:
                #resnet_152_gelu=load_resnet_with_GELU()
                resnet152=load_resnet()
                module_graph = convert_resnet_to_module_graph(resnet152)
                module_graph_ELU=switch_activations_in_dependency_graph(module_graph,torch.nn.ELU)
                return module_graph_ELU
            elif model_nr == 17:
                #resnet_152_gelu=load_resnet_with_GELU()
                resnet152=load_resnet()
                module_graph = convert_resnet_to_module_graph(resnet152)
                module_graph_Tanh=switch_activations_in_dependency_graph(module_graph,torch.nn.Tanh)
                return module_graph_Tanh
            elif model_nr == 18:
                #resnet_152_gelu=load_resnet_with_GELU()
                resnet152=load_resnet()
                module_graph = convert_resnet_to_module_graph(resnet152)
                module_graph_Sigmoid=switch_activations_in_dependency_graph(module_graph,torch.nn.Sigmoid)
                return module_graph_Sigmoid
            return None
            # return [nn.Conv2d(1, 16, 3, 1),
            #         nn.ReLU(),
            #         nn.BatchNorm2d(16),
            #         nn.Conv2d(16, 16, 3, 1),
            #         nn.ReLU(),
            #         nn.BatchNorm2d(16),
            #         nn.MaxPool2d(2),
            #         # nn.Dropout(0.25),
            #         nn.Flatten(),
            #         nn.Linear(2304, 32),
            #         nn.ReLU(),
            #         nn.BatchNorm1d(32),
            #         # nn.Dropout(0.5),
            #         nn.Linear(32, 10),
            #         nn.LogSoftmax(dim=1)]
            # return [nn.Conv2d(1, 32, 3, 1),
            #  nn.ReLU(),
            #  nn.Conv2d(32, 64, 3, 1),
            #  nn.ReLU(),
            #  nn.MaxPool2d(2),
            #  # nn.Dropout(0.25),
            #  nn.Flatten(),
            #  nn.Linear(9216, 64),
            #  nn.ReLU(),
            #  # nn.Dropout(0.5),
            #  nn.Linear(64, 10),
            #  nn.LogSoftmax(dim=1)]
            # return [nn.Conv2d(1, 32, 3, 1),
            #             nn.Tanh(),
            #             nn.Conv2d(32, 64, 3, 1),
            #             nn.Tanh(),
            #             nn.MaxPool2d(2),
            #             # nn.Dropout(0.25),
            #             nn.Flatten(),
            #             nn.Linear(9216, 64),
            #             nn.Tanh(),
            #             # nn.Dropout(0.5),
            #             nn.Linear(64, 10),
            #             nn.LogSoftmax(dim=1)]

        # model_layers=[nn.Conv2d(1, 32, 3, 1),
        #                 nn.ReLU(),
        #                  nn.Conv2d(32, 64, 3, 1),
        #                  nn.ReLU(),
        #                 nn.MaxPool2d(2),
        #                  #nn.Dropout(0.25),
        #                 nn.Flatten(),
        #                  nn.Linear(9216, 64),
        #                  nn.ReLU(),
        #                  #nn.Dropout(0.5),
        #                  nn.Linear(64, 10),
        #                  nn.LogSoftmax(dim=1)]

        # model_layers = [nn.Conv2d(1, 32, 3, 1),
        #                 nn.Tanh(),
        #                 nn.Conv2d(32, 64, 3, 1),
        #                 nn.Tanh(),
        #                 nn.MaxPool2d(2),
        #                 # nn.Dropout(0.25),
        #                 nn.Flatten(),
        #                 nn.Linear(9216, 64),
        #                 nn.Tanh(),
        #                 # nn.Dropout(0.5),
        #                 nn.Linear(64, 10),
        #                 nn.LogSoftmax(dim=1)]
        #lr=0.1

        save_name='results'
        save_name+='_model'+str(model_nr)
        if method==0 or method==20:
            save_name+="_not"
        save_name+="_improved_training"
        if method>1:
            save_name += str(method)
        if method==1 or method==4 or method==5 or method==6 or method==7 or method==8:
            save_name+="_iter"+str(iter_count)
        save_name+="_seed"+str(seed)
        save_name+="_"+str(trainings_same_model)+"trainings"
        save_name += "_" + str(epochs) + "e"
        #save_name+="_const_step"
        #if not mnist:
        save_name+="_"+dataset
        if cross_val:
            save_name+="_cross_validation"+str(folds_num)
        if lr_mul!=1:
            save_name+='_'+str(lr_mul)+'lr'
        if not mini_batch:
            save_name += '_batch'
        if method!=0 and method!=20 and average_gradient_of_nonlinear_layers_enhancement:
            save_name+='__nonlinear_layers_avggrad'
        if method!=0 and method!=20 and average_gradient_of_linear_layers_enhancement:
            save_name+='__linear_layers_avggrad'
        if method!=0 and method!=20 and average_gradient_of_loss:
            save_name+='__loss_avggrad'
        if d_type!=torch.float:
            save_name+='_'+str(d_type)
        if pretraining is not None:
            save_name+='_with_pretraining'+str(pretraining['method'])
        save_name+='_'+optimizer_type.__name__
        if 'betas' in optim_args:
            save_name += '_betas' + str(optim_args['betas']).replace(' ','')
        if 'momentum' in optim_args:
            save_name+='_'+str(optim_args['momentum'])+'momentum'
        save_name+='_batch_size'+str(batch_size)
        save_name+='.txt'
        model=None

        best_stats=None
        processed_stats=None
        actual_stats = None
        #preserve_stats=False

        model_name='model'+str(model_nr)+'_'+(dataset)+'.pt'

        def set_params(model,params):
            #model.gradient_factor = params[0]
            #model.step_factor = params[1]
            #model.gradient_factor_simple_layers = params
            for (param,value) in params.items():
                #if not hasattr(model,param):
                if param not in {'gamma','lr'}:
                    setattr(model,param,value)

        def generate_params():
            global hyperparameter_index
            hyperparameter_index += 1
            # return (rand_between(0,1),rand_between(0,1))
            # return (rand_between(-1, 1)*0.1+0.53, rand_between(-1, 1)*0.1+0.39)
            # return (rand_between(-1, 1)*0.1+0.621, rand_between(-1, 1)*0.1+0.336)
            if optimizer_type == SOAP:
                if 'betas' in optim_args and optim_args['betas'][0]==0.:
                    if dataset=='mnist':
                        if model_nr==6:
                            #lr = 0.0008*1.07**(-2+hyperparameter_index)
                            lr=0.000748
                            # if method!=0 and method!=20:
                            #     lr=0.000748*1.07**hyperparameter_index
                            return {'gamma': float(1), 'lr': float(lr)}
                        elif model_nr==9:
                            #lr = 0.00025*1.22**(-1+hyperparameter_index)
                            lr=0.000824
                            if method != 0 and method != 20:
                                lr = 0.000824*1.22**hyperparameter_index
                            return {'gamma': float(1), 'lr': float(lr)}
                    elif dataset=='fashion_mnist':
                        if model_nr==6:
                            #lr = 0.0015*1.07**(-2+hyperparameter_index)
                            lr=0.0014
                            # if method != 0 and method != 20:
                            #     lr = 0.0014*1.07**hyperparameter_index
                            return {'gamma': float(1), 'lr': float(lr)}
                        elif model_nr==9:
                            #lr = 0.00045*1.22**(-1+hyperparameter_index)
                            lr=0.000817
                            # if method != 0 and method != 20:
                            #     lr = 0.000817*1.22**hyperparameter_index
                            return {'gamma': float(1), 'lr': float(lr)}
                    elif dataset=='imdb' and model_nr==12:
                        #lr = 0.000527*1.15**(-1+hyperparameter_index)
                        lr=0.0014
                        # if method != 0 and method != 20:
                        #     lr = 0.0014*1.15**hyperparameter_index
                        return {'gamma': float(1), 'lr': float(lr)}
                    elif dataset=='imagenet_ood':
                        if model_nr==13:
                            #lr = 1e-4*1.1**(hyperparameter_index)
                            lr=0.000135
                            if method==38 or method == 37 or method == 33:
                                return {'gamma': 1.0, 'lr': lr, 'copy_freq': 300}
                            return {'gamma': float(1), 'lr': float(lr)}
                else:
                    if dataset == 'mnist':
                        if model_nr == 6:
                            #lr = 0.0008 * 1.07 ** (-2 + hyperparameter_index)
                            lr = 0.000748
                            # if method != 0 and method != 20:
                            #     lr = 0.000748 * 1.07 ** hyperparameter_index
                            return {'gamma': float(1), 'lr': float(lr)}
                        elif model_nr == 9:
                            #lr = 0.00025 * 1.22 ** (-1 + hyperparameter_index)
                            lr = 0.000676
                            if method != 0 and method != 20:
                                lr = 0.000676 * 1.22 **hyperparameter_index
                            return {'gamma': float(1), 'lr': float(lr)}
                    elif dataset == 'fashion_mnist':
                        if model_nr == 6:
                            #lr = 0.0015 * 1.07 ** (-2 + hyperparameter_index)
                            lr=0.00161
                            # if method != 0 and method != 20:
                            #     lr =  0.00161*1.07 ** hyperparameter_index
                            return {'gamma': float(1), 'lr': float(lr)}
                        elif model_nr == 9:
                            #lr = 0.00045 * 1.22 ** (-1 + hyperparameter_index)
                            lr=0.000997
                            if method != 0 and method != 20:
                                lr =  0.000997* 1.22 **hyperparameter_index
                            return {'gamma': float(1), 'lr': float(lr)}
                    elif dataset == 'imdb' and model_nr == 12:
                        #lr = 0.000527 * 1.15 ** (-1 + hyperparameter_index)
                        lr=0.00161
                        if method != 0 and method != 20:
                            lr = 0.00161 * 1.15**hyperparameter_index
                        return {'gamma': float(1), 'lr': float(lr)}
                    elif dataset == 'imagenet_ood':
                        if model_nr == 13:
                            #lr = 1e-4 * 1.1 ** (hyperparameter_index)
                            lr=0.000195
                            if method==38 or method == 37 or method == 33:
                                return {'gamma': 1.0, 'lr': lr, 'copy_freq': 300}
                            return {'gamma': float(1), 'lr': float(lr)}
                        elif model_nr == 18:
                            lr = 0.0001*1.3**hyperparameter_index

            if optimizer_type==torch.optim.RMSprop or (optimizer_type==SOAP and 'betas' in optim_args and optim_args['betas'][0]==0.):# or True:
                if 'momentum' not in optim_args:
                    if not mini_batch:
                        if dataset=='mnist':
                            if model_nr==6:
                                #lr=0.0022/1.8**(4./3)#0.0015
                                # lr=0.0012
                                # lr = lr * (1.8) ** (-(hyperparameter_index - (trainings_different_model - 1) / 2) / (
                                #         (trainings_different_model - 1) / 2))
                                lr=0.0016
                                return {'gamma': float(1), 'lr': float(lr)}

                if 'momentum' not in optim_args or method==10:
                    if cross_val:
                        if dataset == 'mnist':
                            if model_nr == 6:
                                # return {'gamma': float(0.9329), 'lr': float(0.02787)}
                                pass
                            elif model_nr == 5:
                                #########return {'gamma': float(0.9145), 'lr': float(0.006728)}
                                # return {'gamma': float(rand_between(0.9145-0.02,0.9145+0.04)), 'lr': float(0.006728*0.7**rand_between(-1,1))}
                                # return {'gamma': float(rand_between(0.9145-0.01,0.9145+0.05)), 'lr': float(0.003297*0.7**rand_between(-1,1))}
                                # return {'gamma': float(rand_between(0.9145,0.9145+0.06)), 'lr': float(0.001615*0.7**rand_between(-1,1))}
                                # return {'gamma': float(rand_between(0.9145+0.01,0.9145+0.07)), 'lr': float(0.0007915*0.7**rand_between(-1,1))}
                                # return {'gamma': float(rand_between(0.915+0.025,1)), 'lr': float(0.0003878*0.7**rand_between(-1,1))}

                                # return {'gamma': float(rand_between(0.8848, 0.9048)),'lr': float(0.002897 * 0.9 ** rand_between(-2, 0))}#6*100
                                # return {'gamma': float(rand_between(0.8848, 0.9048)),'lr': float(0.002897 * 0.9 ** rand_between(0, 2))}#6*100
                                return {'gamma': float(rand_between(0.8649, 0.8849)),
                                        'lr': float(0.003098 * 0.9 ** rand_between(-1.7, 0.3))}
                        elif dataset == 'fashion_mnist':
                            if model_nr == 6:
                                # return {'gamma': float(rand_between(0.955 - 0.01, 0.955 + 0.01)),'lr': float(0.04535 * (0.9 ** rand_between(-1, 1)))}
                                # return {'gamma': float(0.9581), 'lr': float(0.04678)}
                                pass
                            elif model_nr == 5:
                                # return {'gamma': float(0.9384), 'lr': float(0.008208)}
                                pass
                    else:
                        if dataset == 'mnist':
                            if model_nr == 10:
                                # lr = (0.00005 + 0.00005 * hyperparameter_index)
                                # lr = 0.0005  # temporarily
                                lr = 0.00055
                                return {'gamma': float(1), 'lr': float(lr)}
                            elif model_nr == 9:
                                # # lr=(0.001+0.001*hyperparameter_index) if hyperparameter_index<=10 else (0.0001+0.0001*(hyperparameter_index-11))
                                # lr=(0.00005+0.00005*hyperparameter_index) if hyperparameter_index<=8 else (0.0005+0.0001*(hyperparameter_index-9))
                                # if method>0 and separate_hyperparameters_for_optimized:
                                #     # lr=0.0003+0.0003*hyperparameter_index if hyperparameter_index<=4 else (0.002+0.001*(hyperparameter_index-5))
                                #     lr = 0.0009 + 0.0006 * hyperparameter_index if hyperparameter_index <= 1 else (
                                #             0.002 + 0.001 * (hyperparameter_index - 2))
                                # return {'gamma': float(1), 'lr': float(lr)}

                                # if method>0 and separate_hyperparameters_for_optimized:
                                #     if method == 1:
                                #         if iter_count == 2:
                                #             return {'gamma': float(1), 'lr': float(0.0015)}
                                #         elif iter_count > 2:
                                #             return {'gamma': float(1), 'lr': float(0.0015)}#exception
                                if method > 0 and separate_hyperparameters_for_optimized:
                                    if method >= 1:
                                        # if iter_count == 2:
                                        #     return {'gamma': float(1), 'lr': float(0.00075)}
                                        # elif iter_count > 2:
                                        #     return {'gamma': float(1), 'lr': float(0.00075)}

                                        # lr = (0.0005 + 0.000125 * hyperparameter_index)
                                        # lr = 0.000625
                                        # #return {'gamma': float(1), 'lr': float(lr)}
                                        # return {'gamma': float(1),
                                        #         'lr': float(lr * np.e ** ((hyperparameter_index - 4) / (4) * (np.log(3))))}
                                        # return {'gamma': float(1),
                                        #         'lr': float([0.00006, 0.00008, 0.00009, 0.0001, 0.00011, 0.00012, 0.00014][
                                        #                         hyperparameter_index])}
                                        lr = 0.0009
                                        if iter_count>2:
                                            if method == 22:
                                                return {'gamma': 1.0, 'lr': float(lr),
                                                        'gradient_factor2': 2. / iter_count}
                                            elif method==23:
                                                return {'gamma': 1.0, 'lr': float(lr),
                                                        'gradient_factor2': 2. / iter_count}
                                        return {'gamma': float(1), 'lr': float(lr)}

                                # return {'gamma': float(1), 'lr': float(0.00025)}
                                # lr = (0.00015 + 0.00005 * hyperparameter_index)
                                # lr = 0.00035
                                # #return {'gamma': float(1), 'lr': float(lr)}
                                # return {'gamma': float(1), 'lr': float(lr * np.e ** ((hyperparameter_index - 4) / (4) * (np.log(3))))}
                                # return {'gamma': float(1),
                                #         'lr': float([0.00015, 0.0002, 0.00025,0.000275, 0.0003, 0.00035, 0.0004][
                                #                         hyperparameter_index])}
                                lr = 0.00025
                                return {'gamma': float(1), 'lr': float(lr)}
                            elif model_nr == 8:
                                return {'gamma': float(1), 'lr': float(0.001)}
                            elif model_nr == 7:
                                # if hyperparameter_index < 3:
                                #     return {'gamma': float(1), 'lr': float(0.00005 + hyperparameter_index * 0.00005)}
                                # return {'gamma': float(1), 'lr': float(0.0002 + (hyperparameter_index - 3) * 0.0001)}
                                # return {'gamma': float(1), 'lr': float(0.0009)}
                                return {'gamma': float(1), 'lr': float(0.0001 + hyperparameter_index * 0.0001)}

                            elif model_nr == 6:
                                # return {'gamma':float(rand_between(0.9, 1)),'lr':float(0.1**rand_between(1,3))}
                                # return {'gamma':float(rand_between(0.906-0.1, 0.906+0.05)),'lr':float(0.0152*(0.3**rand_between(-1,1)))}
                                ###################
                                # return {'gamma':float(rand_between(0.908-0.05, 0.908+0.05)),'lr':float(0.0202*(0.7**rand_between(-1,1)))}
                                # return {'gamma':float(rand_between(0.925-0.01, 0.925+0.01)),'lr':float(0.02115*(0.95**rand_between(-1,1)))}
                                # return {'gamma':float(rand_between(0.929-0.01, 0.929+0.01)),'lr':float(0.02204*(0.9**rand_between(-1,0)))}
                                # return {'gamma': float(rand_between(0.933 - 0.01, 0.933 + 0.01)),'lr': float(0.02424 * (0.9 ** rand_between(-1, 0)))}
                                # return {'gamma': float(rand_between(0.937 - 0.01, 0.937 + 0.01)),'lr': float(0.02666 * (0.9 ** rand_between(-1, 0)))}
                                # return {'gamma': float(0.9329), 'lr': float(0.02787)}
                                # lr = (0.001 + 0.0005 * hyperparameter_index)
                                # return {'gamma': float(1), 'lr': float(lr)}
                                # if method > 0 and separate_hyperparameters_for_optimized:
                                #     lr = 0.0008 + 0.0004 * hyperparameter_index
                                #     return {'gamma': float(1), 'lr': float(lr)}
                                if method >= 1:
                                    if iter_count == 2:
                                        return {'gamma': float(1), 'lr': float(0.0008)}
                                    elif iter_count > 2:
                                        if method == 22:
                                            return {'gamma': 1.0, 'lr': float(0.0008),
                                                    'gradient_factor2': 2. / iter_count}
                                        elif method==23:
                                            return {'gamma': 1.0, 'lr': float(0.0008),
                                                    'gradient_factor2': 2. / iter_count}
                                        return {'gamma': float(1), 'lr': float(0.0008)}

                                return {'gamma': float(1), 'lr': float(0.0008)}
                            elif model_nr == 5:
                                if method == 1 and separate_hyperparameters_for_optimized:
                                    # return {'gamma': float(1), 'lr': float(0.001*0.3**rand_between(-1,1))}
                                    return {'gamma': float(1), 'lr': float(0.001023 * 0.9 ** rand_between(-1, 1))}

                                # return {'gamma':float(rand_between(0.906-0.1, 0.906+0.05)),'lr':float(0.0152*(0.3**rand_between(-1,1)))}
                                # return {'gamma':float(rand_between(0.896-0.05, 0.896+0.05)),'lr':float(0.00514*(0.3**rand_between(-0.15,1)))}
                                ####################
                                # return {'gamma':float(rand_between(0.896-0.05, 0.896+0.05)),'lr':float(0.00514*(0.7**rand_between(-1,1)))}
                                # return {'gamma':float(rand_between(0.9182-0.01, 0.9182+0.01)),'lr':float(0.00654*(0.95**rand_between(-1,1)))}
                                # return {'gamma':float(0.9145),'lr':float(0.006728)}
                                # return {'gamma': float(1), 'lr': float(0.00009 + hyperparameter_index * 0.00001)}
                                return {'gamma': float(1), 'lr': float(0.0001 + hyperparameter_index * 0.0001)}

                        elif dataset == 'fashion_mnist':
                            if model_nr == 10:
                                # lr = (0.00005 + 0.00005 * hyperparameter_index)
                                # lr = 0.0005#temporarily
                                lr = 0.00055
                                return {'gamma': float(1), 'lr': float(lr)}
                            elif model_nr == 9:
                                # lr = (0.00005 + 0.00005 * hyperparameter_index) if hyperparameter_index <= 8 else (
                                #             0.0005 + 0.0001 * (hyperparameter_index - 9))
                                # if method > 0 and separate_hyperparameters_for_optimized:
                                #     # lr = 0.0003 + 0.0003 * hyperparameter_index if hyperparameter_index <= 4 else (
                                #     #             0.002 + 0.001 * (hyperparameter_index - 5))
                                #     lr = 0.0015 + 0.0005 * hyperparameter_index if hyperparameter_index <= 3 else (
                                #             0.004 + 0.001 * (hyperparameter_index - 4))
                                # return {'gamma': float(1), 'lr': float(lr)}
                                if method > 0 and separate_hyperparameters_for_optimized:
                                    if method >= 1:
                                        # if iter_count == 2:
                                        #     return {'gamma': float(1), 'lr': float(0.0009)}
                                        # elif iter_count > 2:
                                        #     return {'gamma': float(1), 'lr': float(0.0009)}

                                        # lr = (0.0006 + 0.00015 * hyperparameter_index)
                                        # lr = 0.00105
                                        # #return {'gamma': float(1), 'lr': float(lr)}
                                        # return {'gamma': float(1), 'lr': float(lr * np.e ** ((hyperparameter_index - 4) / (4) * (np.log(3))))}
                                        #return {'gamma': float(1), 'lr': float([0.0006,0.0008,0.0009,0.001,0.0011,0.0012,0.0014][hyperparameter_index])}
                                        lr = 0.0012
                                        if method == 22:
                                            return {'gamma': 1.0, 'lr': float(lr),
                                                    'gradient_factor2': 2. / iter_count}
                                        elif method==23:
                                            return {'gamma': 1.0, 'lr': float(lr),
                                                    'gradient_factor2': 2. / iter_count}
                                        return {'gamma': float(1), 'lr': float(lr)}
                                # return {'gamma': float(1), 'lr': float(0.0003)}
                                # lr = (0.0002 + 0.00005 * hyperparameter_index)
                                # lr = 0.0004
                                # #return {'gamma': float(1), 'lr': float(lr)}
                                # return {'gamma': float(1), 'lr': float(lr * np.e ** ((hyperparameter_index - 4) / (4) * (np.log(3))))}
                                # return {'gamma': float(1),
                                #         'lr': float([0.0003,0.00035,0.0004, 0.00045, 0.0005, 0.00055, 0.0006][hyperparameter_index])}
                                #lr = 0.00035
                                lr=0.00045#average of two very good learning rates: 0.00035 and 0.00055
                                return {'gamma': float(1), 'lr': float(lr)}
                            if model_nr == 8:
                                return {'gamma': float(1), 'lr': float(0.001)}
                            elif model_nr == 7:
                                # if hyperparameter_index<3:
                                #     return {'gamma': float(1), 'lr': float(0.00005+hyperparameter_index*0.00005)}
                                # return {'gamma': float(1), 'lr': float(0.0002 + (hyperparameter_index-3) * 0.0001)}
                                # return {'gamma': float(1), 'lr': float(0.0002)}
                                return {'gamma': float(1), 'lr': float(0.00002 + hyperparameter_index * 0.00003)}

                            elif model_nr == 6:
                                # return {}
                                # return {'gamma': float(rand_between(0.937 - 0.03, 0.937 + 0.03)),'lr': float(0.02666 * (0.7 ** rand_between(-2, 0)))}
                                # return {'gamma': float(rand_between(0.955 - 0.01, 0.955 + 0.01)),'lr': float(0.04535 * (0.9 ** rand_between(-1, 1)))}
                                ############## fixed
                                # return {'gamma': float(0.9581), 'lr': float(0.04678)}
                                # lr = (0.001 + 0.0005 * hyperparameter_index)
                                # return {'gamma': float(1), 'lr': float(lr)}
                                if method > 0 and separate_hyperparameters_for_optimized:
                                    # lr = 0.0015 + 0.0005 * hyperparameter_index
                                    # return {'gamma': float(1), 'lr': float(lr)}
                                    if method >= 1:
                                        if iter_count == 2:
                                            return {'gamma': float(1), 'lr': float(0.0019)}
                                        elif iter_count > 2:
                                            if method == 22:
                                                return {'gamma': 1.0, 'lr': float(0.0019),
                                                        'gradient_factor2': 2. / iter_count}
                                            elif method==23:
                                                return {'gamma': 1.0, 'lr': float(0.0019),
                                                        'gradient_factor2': 2. / iter_count}
                                            return {'gamma': float(1), 'lr': float(0.0015)}

                                return {'gamma': float(1), 'lr': float(0.0015)}

                            elif model_nr == 5:
                                # return {'gamma': float(rand_between(0.9145 - 0.03, 0.9145 + 0.03)),'lr': float(0.006728 * (0.7 ** rand_between(-1, 1)))}
                                # return {'gamma': float(rand_between(0.930 - 0.01, 0.930 + 0.01)),'lr': float(0.00750 * (0.9 ** rand_between(-1, 1)))}
                                # return {'gamma': float(rand_between(0.939 - 0.01, 0.939 + 0.01)),'lr': float(0.008130 * (0.9 ** rand_between(-1, 0)))}
                                # return {'gamma': float(0.9384), 'lr': float(0.008208)}

                                # return {'gamma': float(1), 'lr': float(0.0001)}
                                # return {'gamma': float(1), 'lr': float(0.0001*3**rand_between(-1,1))}
                                # return {'gamma': float(1), 'lr': float(0.00009+hyperparameter_index*0.00001)}
                                return {'gamma': float(1), 'lr': float(0.0001 + hyperparameter_index * 0.0001)}
                        elif dataset=='imdb':
                            if model_nr==11:
                                if method >= 1:
                                    if iter_count == 2:
                                        return {'gamma': float(1), 'lr': float(0.0004)}
                                    elif iter_count > 2:
                                        return {'gamma': float(1), 'lr': float(0.0004)}
                                #return {'gamma': float(1), 'lr': float(0.0001)}
                                #return {'gamma': float(1), 'lr': float(0.0005)}
                                #return {'gamma': float(1), 'lr': float(0.00003**(1+hyperparameter_index/(10-1)*(np.log(0.0003)/np.log(0.00003)-1)))}
                                return {'gamma': float(1), 'lr': float(0.0004)}
                            if model_nr==12:
                                if method >= 1:
                                    if iter_count == 2:
                                        #return {'gamma': float(1), 'lr': float(0.0001)}
                                        #return {'gamma': float(1), 'lr': float(0.0025 ** (1 + (hyperparameter_index - 4) / (4) * (np.log(0.0003) / np.log(0.0001) - 1)))}
                                        #return {'gamma': float(1), 'lr': float(0.0003641*1.3 ** (hyperparameter_index+1))}
                                        #return {'gamma': 1.0, 'lr': 0.00047333}
                                        # return {'gamma': 1.0, 'lr': 0.0006906444697974225}#outlier of method 0
                                        return {'gamma': 1.0, 'lr': 0.000527}#the same as for gradient RMSProp
                                    elif iter_count > 2:
                                        #return {'gamma': float(1), 'lr': float(0.0001)}
                                        #return {'gamma': float(1), 'lr': float(0.0025 ** (1 + (hyperparameter_index - 4) / (4) * (np.log(0.0003) / np.log(0.0001) - 1)))}
                                        #return {'gamma': float(1), 'lr': float(0.0003641 * 1.3 ** (hyperparameter_index + 1))}
                                        #return {'gamma': 1.0, 'lr': 0.00047333}
                                        if method==22:
                                            return {'gamma': 1.0, 'lr': 0.000527,'gradient_factor2': 2./iter_count}
                                        elif method==23:
                                            return {'gamma': 1.0, 'lr': 0.000527, 'gradient_factor2': 2. / iter_count}
                                        return {'gamma': 1.0, 'lr': 0.000527}#the same as for gradient RMSProp
                                        # return {'gamma': 1.0, 'lr': 0.0006906444697974225}#outlier of method 0
                                #return {'gamma': float(1), 'lr': float(0.0001)}
                                #return {'gamma': float(1), 'lr': float(0.0015 ** (1 + (hyperparameter_index - 4) / (4) * (np.log(0.0003) / np.log(0.0001) - 1)))}
                                #return {'gamma': float(1), 'lr': float(0.0005689123460087089*1.25**-hyperparameter_index)}
                                #return {'gamma': 1.0, 'lr': 0.0006906444697974225}#outlier
                                #return {'gamma': 1.0, 'lr': 0.0003641}
                                return {'gamma': 1.0, 'lr': 0.000527}#average of 0.0003641 and 0.0006906
                        elif dataset == 'imagenet_ood':
                            if model_nr == 13:# or model_nr==14:
                                #lr = 7.5e-5
                                #lr = 6e-5+1e-5*hyperparameter_index
                                lr=1e-4
                                if method == 22:
                                    # lr = 1e-4
                                    # return {'gamma': 1.0, 'lr': lr,'gradient_factor2': float(hyperparameter_index*0.005+0.005)}
                                    # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 1.}# 0.025}
                                    #return {'gamma': 1.0, 'lr': 0.0001}
                                    return {'gamma': 1.0, 'lr': lr}
                                    return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.1+0.1*hyperparameter_index}
                                elif method == 23:
                                    # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5/(iter_count/2)**0.5}

                                    # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5/(iter_count/2)**0.5,'gradient_factor2_decrease_power':75./25.}
                                    # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5 / (iter_count / 2) ** 0.5,
                                    #         'gradient_factor2_decrease_power':
                                    #             [80. / 20, 90. / 10, 95. / 5, 80. / 20, 90. / 10, 95. / 5, 80. / 20,
                                    #              90. / 10,
                                    #              95. / 5, 80. / 20, 90. / 10,
                                    #              95. / 5][hyperparameter_index],
                                    #         'max_gradient_factor2':
                                    #             [1., 1., 1., 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.15, 0.15, 0.15][
                                    #                 hyperparameter_index]}  # 0.05+0.05*hyperparameter_index}
                                    return {'gamma': 1.0, 'lr': 0.0001, 'gradient_factor2': 0.5,
                                            'gradient_factor2_decrease_power': 4.0, 'max_gradient_factor2': 0.5}
                                elif method==38 or method==37 or method==33:
                                    #return {'gamma': 1.0, 'lr': lr, 'copy_freq': 20+15*hyperparameter_index}
                                    return {'gamma': 1.0, 'lr': lr, 'copy_freq': 35}
                                return {'gamma': 1.0, 'lr': lr}
                            elif model_nr==14:
                                #lr = 7.5e-5
                                #lr = 6e-5+1e-5*hyperparameter_index
                                lr=1e-4
                                if method == 22:
                                    # lr = 1e-4
                                    # return {'gamma': 1.0, 'lr': lr,'gradient_factor2': float(hyperparameter_index*0.005+0.005)}
                                    # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 1.}# 0.025}
                                    return {'gamma': 1.0, 'lr': 0.0001}
                                elif method == 23:
                                    # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5/(iter_count/2)**0.5}

                                    # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5/(iter_count/2)**0.5,'gradient_factor2_decrease_power':75./25.}
                                    # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5 / (iter_count / 2) ** 0.5,
                                    #         'gradient_factor2_decrease_power':
                                    #             [80. / 20, 90. / 10, 95. / 5, 80. / 20, 90. / 10, 95. / 5, 80. / 20,
                                    #              90. / 10,
                                    #              95. / 5, 80. / 20, 90. / 10,
                                    #              95. / 5][hyperparameter_index],
                                    #         'max_gradient_factor2':
                                    #             [1., 1., 1., 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.15, 0.15, 0.15][
                                    #                 hyperparameter_index]}  # 0.05+0.05*hyperparameter_index}
                                    return {'gamma': 1.0, 'lr': 0.0001, 'gradient_factor2': 0.5,
                                            'gradient_factor2_decrease_power': 4.0, 'max_gradient_factor2': 0.5}
                                return {'gamma': 1.0, 'lr': lr}
                            elif model_nr == 15:# or model_nr==14:
                                #lr = 7.5e-5
                                #lr = 6e-5+1e-5*hyperparameter_index
                                #lr=1e-4
                                #lr = [8e-5, 9e-5, 1e-4, 1.1e-4, 1.2e-4, 1.3e-4][hyperparameter_index]
                                lr=1.5e-4
                                if method == 22:
                                    # lr = 1e-4
                                    #return {'gamma': 1.0, 'lr': lr,'gradient_factor2': float(hyperparameter_index*0.005+0.005)}
                                    #return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 1.}# 0.025}
                                    return {'gamma': 1.0, 'lr': lr}
                                elif method == 23:
                                    #return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5/(iter_count/2)**0.5}

                                    #return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5/(iter_count/2)**0.5,'gradient_factor2_decrease_power':75./25.}
                                    # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5 / (iter_count / 2) ** 0.5,
                                    #         'gradient_factor2_decrease_power':
                                    #             [80. / 20, 90. / 10, 95. / 5, 80. / 20, 90. / 10, 95. / 5, 80. / 20,
                                    #              90. / 10,
                                    #              95. / 5, 80. / 20, 90. / 10,
                                    #              95. / 5][hyperparameter_index],
                                    #         'max_gradient_factor2':
                                    #             [1., 1., 1., 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.15, 0.15, 0.15][
                                    #                 hyperparameter_index]}  # 0.05+0.05*hyperparameter_index}
                                    return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5, 'gradient_factor2_decrease_power': 4.0, 'max_gradient_factor2': 0.5}
                                return {'gamma': 1.0, 'lr': lr}
                            elif model_nr == 16:
                                lr = 1.2e-4
                                return {'gamma': 1.0, 'lr': lr}
                            elif model_nr == 17:
                                lr = 1.2e-4
                                return {'gamma': 1.0, 'lr': lr}
                            elif model_nr == 18:
                                # lr = [8e-5, 9e-5, 1e-4, 1.1e-4, 7e-5, 1.2e-4, 1.3e-4, 6e-5, 1.4e-4,5e-5,4e-5][
                                #     hyperparameter_index]
                                #lr = 8e-5-hyperparameter_index*1e-5
                                lr=1e-5
                                return {'gamma': 1.0, 'lr': lr}
                    #return {'gamma': float(1), 'lr': float(0.0001)}
                elif 'momentum' in optim_args:
                    if dataset == 'mnist':
                        if model_nr == 6:
                            lr = 0.000101
                            lr=0.000101*(1.8**(-7./3))
                            lr=lr*(1.8) ** (-(hyperparameter_index - (trainings_different_model - 1) / 2) / ((trainings_different_model - 1) / 2))
                            return {'gamma': 1.0, 'lr': lr}
                        elif model_nr == 9:
                            lr = 0.0000258
                            lr=0.0000258*(1.8**(-7./3))
                            # lr = lr * (1.8) ** ((hyperparameter_index - (trainings_different_model - 1) / 2) / (
                            #             (trainings_different_model - 1) / 2))
                            lr = lr * (1.8) ** (-(hyperparameter_index - (trainings_different_model - 1) / 2) / (
                                        (trainings_different_model - 1) / 2))
                            return {'gamma': 1.0, 'lr': lr}
                    elif dataset == 'fashion_mnist':
                        if model_nr == 6:
                            lr = 0.000189
                            lr = 0.000189 * (1.8) ** (-7. / 3.)
                            lr = lr * (1.8) ** (-(hyperparameter_index - (trainings_different_model - 1) / 2) / (
                                        (trainings_different_model - 1) / 2))
                            return {'gamma': 1.0, 'lr': lr}
                        elif model_nr == 9:
                            lr = 0.0000298
                            lr = 0.0000298 * (1.8) ** (-7. / 3)#0.000007561
                            # lr = lr * (1.8) ** ((hyperparameter_index - (trainings_different_model - 1) / 2) / (
                            #             (trainings_different_model - 1) / 2))
                            lr = lr * (1.8) ** (-(hyperparameter_index - (trainings_different_model - 1) / 2) / (
                                    (trainings_different_model - 1) / 2))
                            return {'gamma': 1.0, 'lr': lr}
                    elif dataset == 'imdb':
                        if model_nr == 12:
                            lr = 0.0000255
                            # lr=0.0000558346
                            lr=0.0000255*(1.8**(7./3))
                            lr = lr * (1.8) ** ((hyperparameter_index - (trainings_different_model - 1) / 2) / (
                                        (trainings_different_model - 1) / 2))
                            return {'gamma': 1.0, 'lr': lr}
            elif optimizer_type==optim.Adam or optimizer_type == SOAP:
                if not mini_batch:
                    if dataset=='mnist':
                        lr=0.0015
                        lr = lr * (1.8) ** (-(hyperparameter_index - (trainings_different_model - 1) / 2) / (
                                (trainings_different_model - 1) / 2))
                        return {'gamma': 1.0, 'lr': lr}
                if dataset=='mnist':
                    if model_nr==6:
                        #lr = 3.*0.0008 * (1.8) ** ((hyperparameter_index - (trainings_different_model - 1) / 2) / ((trainings_different_model - 1) / 2))
                        #lr = 3.*0.0008 * (1.8) ** ((hyperparameter_index - (trainings_different_model - 1) / 2) / (
                        #        (trainings_different_model - 1) / 2))*(1.8**(-2.-1./7))
                        lr=0.00101
                        return {'gamma': 1.0, 'lr': lr}
                    elif model_nr==9:
                        #lr = 3.*0.00025 * (1.8) ** ((hyperparameter_index - (trainings_different_model - 1) / 2) / ((trainings_different_model - 1) / 2))
                        #lr = 3.*0.00025 * (1.8) ** ((hyperparameter_index - (trainings_different_model - 1) / 2) / (
                        #        (trainings_different_model - 1) / 2))*(1.8**(-2.-1./7))
                        lr=0.000258
                        return {'gamma': 1.0, 'lr': lr}
                elif dataset=='fashion_mnist':
                    if model_nr==6:
                        #lr = 3.*0.0015 * (1.8) ** ((hyperparameter_index - (trainings_different_model - 1) / 2) / ((trainings_different_model - 1) / 2))
                        #lr = 3.*0.0015 * (1.8) ** ((hyperparameter_index - (trainings_different_model - 1) / 2) / (
                        #        (trainings_different_model - 1) / 2))*(1.8**(-2.-1./7))
                        lr=0.00189
                        return {'gamma': 1.0, 'lr': lr}
                    elif model_nr==9:
                        #lr = 3.*0.00035 * (1.8) ** ((hyperparameter_index - (trainings_different_model - 1) / 2) / ((trainings_different_model - 1) / 2))
                        #lr = 3.*0.00035 * (1.8) ** ((hyperparameter_index - (trainings_different_model - 1) / 2) / (
                        #            (trainings_different_model - 1) / 2))*(1.8**(-2.-1./7))
                        lr=0.000298
                        return {'gamma': 1.0, 'lr': lr}
                elif dataset=='imdb':
                    if model_nr==12:
                        #lr=3.*0.0003641*(1.8)**((hyperparameter_index-(trainings_different_model-1)/2)/((trainings_different_model-1)/2))
                        #lr=3.*0.0003641*(1.8)**((hyperparameter_index-(trainings_different_model-1)/2)/((trainings_different_model-1)/2))*(1.8**(-2.-1./7))
                        lr=0.000255
                        return {'gamma': 1.0, 'lr': lr}
                elif dataset=='imagenet_ood':
                    if model_nr==13:# or model_nr==14:
                        #lr=7.5e-5
                        #lr = 6e-5 + 1e-5 * hyperparameter_index
                        lr=1e-4
                        if method == 22:
                            return {'gamma': 1.0, 'lr': lr}
                            return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.1 + 0.1 * hyperparameter_index}
                        if method == 23:
                            # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5/(iter_count/2)**0.5}
                            # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5/(iter_count/2)**0.5,'gradient_factor2_decrease_power':[95./0.5,9./1.,85./15.,80./20.,75./25.,70./30.,65./35.,60./40.][hyperparameter_index]}
                            # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5 / (iter_count / 2) ** 0.5,
                            #         'gradient_factor2_decrease_power':
                            #             [80. / 20, 90. / 10, 95. / 5, 80. / 20, 90. / 10, 95. / 5, 80. / 20, 90. / 10,
                            #              95. / 5, 80. / 20, 90. / 10,
                            #              95. / 5][hyperparameter_index],
                            #         'max_gradient_factor2': [1., 1., 1., 0.5, 0.5, 0.5, 0.25, 0.25, 0.25,0.15,0.15,0.15][
                            #             hyperparameter_index]}  # 0.05+0.05*hyperparameter_index}
                            return {'gamma': 1.0, 'lr': 0.0001, 'gradient_factor2': 0.5, 'gradient_factor2_decrease_power': 4.0, 'max_gradient_factor2': 1.0}
                        elif method==38 or method == 37 or method == 33:
                            #return {'gamma': 1.0, 'lr': lr, 'copy_freq': 200 + 100 * hyperparameter_index}
                            return {'gamma': 1.0, 'lr': lr, 'copy_freq': 300}

                        return {'gamma': 1.0, 'lr': lr}
                    elif model_nr==14:
                        #lr = 6e-5 + 1e-5 * hyperparameter_index
                        lr = 1.1e-4
                        if method == 23:
                            # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5/(iter_count/2)**0.5}
                            # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5/(iter_count/2)**0.5,'gradient_factor2_decrease_power':[95./0.5,9./1.,85./15.,80./20.,75./25.,70./30.,65./35.,60./40.][hyperparameter_index]}
                            # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5 / (iter_count / 2) ** 0.5,
                            #         'gradient_factor2_decrease_power':
                            #             [80. / 20, 90. / 10, 95. / 5, 80. / 20, 90. / 10, 95. / 5, 80. / 20, 90. / 10,
                            #              95. / 5, 80. / 20, 90. / 10,
                            #              95. / 5][hyperparameter_index],
                            #         'max_gradient_factor2': [1., 1., 1., 0.5, 0.5, 0.5, 0.25, 0.25, 0.25,0.15,0.15,0.15][
                            #             hyperparameter_index]}  # 0.05+0.05*hyperparameter_index}
                            return {'gamma': 1.0, 'lr': 0.0001, 'gradient_factor2': 0.5,
                                    'gradient_factor2_decrease_power': 4.0, 'max_gradient_factor2': 1.0}

                        return {'gamma': 1.0, 'lr': lr}
                    elif model_nr==15:
                        #lr = 6e-5 + 1e-5 * hyperparameter_index
                        lr = 1e-4
                        #lr = [8e-5, 9e-5, 1e-4, 1.1e-4, 1.2e-4, 1.3e-4][hyperparameter_index]
                        lr = 1.7e-4
                        if method == 23:
                            # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5/(iter_count/2)**0.5}
                            # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5/(iter_count/2)**0.5,'gradient_factor2_decrease_power':[95./0.5,9./1.,85./15.,80./20.,75./25.,70./30.,65./35.,60./40.][hyperparameter_index]}
                            # return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5 / (iter_count / 2) ** 0.5,
                            #         'gradient_factor2_decrease_power':
                            #             [80. / 20, 90. / 10, 95. / 5, 80. / 20, 90. / 10, 95. / 5, 80. / 20, 90. / 10,
                            #              95. / 5, 80. / 20, 90. / 10,
                            #              95. / 5][hyperparameter_index],
                            #         'max_gradient_factor2': [1., 1., 1., 0.5, 0.5, 0.5, 0.25, 0.25, 0.25,0.15,0.15,0.15][
                            #             hyperparameter_index]}  # 0.05+0.05*hyperparameter_index}
                            return {'gamma': 1.0, 'lr': lr, 'gradient_factor2': 0.5,
                                    'gradient_factor2_decrease_power': 4.0, 'max_gradient_factor2': 1.0}

                        return {'gamma': 1.0, 'lr': lr}
                    elif model_nr==16:
                        lr = 1.2e-4
                        return {'gamma': 1.0, 'lr': lr}
                    elif model_nr==17:
                        lr = 1.2e-4
                        return {'gamma': 1.0, 'lr': lr}
                    elif model_nr==18:
                        lr = [8e-5, 9e-5, 1e-4, 1.1e-4, 7e-5, 1.2e-4, 1.3e-4, 6e-5, 1.4e-4, 5e-5, 4e-5][
                            hyperparameter_index]
                        return {'gamma': 1.0, 'lr': lr}


        summary=[]

        backup_name='_backup_'+save_name

        state=read_state(backup_name)
        trainings_different_model_start, trainings_same_model_start=(0,0)
        if state:
            global hyperparameter_index
            hyperparameter_index,actual_stats,summary,trainings_different_model_start,trainings_same_model_start,r_state=read_state(backup_name)
            torch.set_rng_state(r_state)

        model = None
        model2 = None
        #scheduler = None
        global model3
        optimizer = None
        opt2 = None
        global opt3
        for _test_params in range(trainings_different_model_start,trainings_different_model):

            # if model is not None:
            #     model = model.cpu()
            # if model2 is not None:
            #     model2 = model2.cpu()
            # if model3 is not None:
            #     model3 = model3.cpu()
            if hasattr(model, 'break_dependency_graph'):
                model.break_dependency_graph()
            if hasattr(model2, 'break_dependency_graph'):
                model2.break_dependency_graph()
            if hasattr(model3, 'break_dependency_graph'):
                model3.break_dependency_graph()
                # import gc
            if model is not None:
                model_ = model
                model = None
                del model_
            if model2 is not None:
                model2_ = model2
                model2 = None
                del model2_
            if model3 is not None:
                model3_ = model3
                model3 = None
                del model3_
            # if scheduler is not None:
            #     scheduler_=scheduler
            #     scheduler=None
            #     del scheduler_
            if optimizer is not None:
                optimizer_ = optimizer
                optimizer = None
                del optimizer_
            if opt2 is not None:
                opt2_ = opt2
                opt2 = None
                del opt2_

            torch.cuda.empty_cache()
            gc.collect()
            print('Initial memory: ' + str(torch.cuda.memory_allocated()))


            params=generate_params()
            params['lr']=lr_mul*params['lr']
            if trainings_same_model_start==0:
                actual_stats= {}
            for _test_same_model in range(trainings_same_model_start,trainings_same_model):
                trainings_same_model_start=0
                #model = NN(copy.deepcopy(model_layers)).to(device)

                if cross_val:
                    train_loader=None
                    #test_loader=None

                    skf=StratifiedKFold(n_splits=folds_num,shuffle=True,random_state=1)
                    for i, (train_index, val_index) in enumerate(skf.split(dataset1.train_data,dataset1.train_labels)):
                        print("Test number [of different models . of the same model . split]: " + str(_test_params) + "." + str(
                            _test_same_model)+"."+str(i))
                        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
                        val_subsampler = torch.utils.data.SubsetRandomSampler(val_index)
                        train_loader = torch.utils.data.DataLoader(
                            dataset1,
                            batch_size=batch_size, sampler=train_subsampler)
                        val_loader = torch.utils.data.DataLoader(
                            dataset1,
                            batch_size=batch_size, sampler=val_subsampler)

                        assert model_layer_graph or method == 4
                        model = NN_Residual(get_layers()).to(device) if model_layer_graph else (
                            NN_Advanced(get_layers()).to(device) if method == 4 else NN(get_layers()).to(device))
                        if model_load:
                            model=load_model(model_name)
                        print("Params: " + str(params))

                        model.stats = actual_stats
                        model.stats['params'] = params
                        set_params(model, params)

                        lr = params['lr']
                        #optimizer = optim.RMSprop(model.parameters(), lr)
                        optimizer=optimizer_type(model.parameters(), lr,**optim_args)

                        gamma = params['gamma']
                        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

                        # model2 = NN_Advanced(copy.deepcopy(model.layers)) if method==4 else NN(copy.deepcopy(model.layers))#NN(copy.deepcopy(model.layers)).to(device)
                        # model2.train()
                        # model2.load_state_dict(model.state_dict())
                        model2=copy.deepcopy(model)
                        # opt2 = type(optimizer)(model2.parameters(),lr=0.01,momentum=0,nesterov=False)
                        # opt2 = type(optimizer)(model2.parameters())
                        opt2 = type(optimizer)(model2.parameters(), lr=optimizer.param_groups[0]['lr'],**optim_args)
                        opt2.load_state_dict(optimizer.state_dict())

                        for epoch in range(1, epochs + 1):
                            if mini_batch:
                                train_minibatch(model, device, train_loader, optimizer, epoch, model2, opt2,log_interval=log_interval)
                            else:
                                train(model, device, train_loader, optimizer, epoch, model2, opt2,log_interval=log_interval)
                            # train(args, model, device, train_loader, optimizer, epoch)#train(args, model, device, train_loader, optimizer, epoch,model2,opt2)
                            test(model, device, test_loader, train_loader,val_loader=val_loader, epoch=epoch)
                            scheduler.step()


                            actual_stats = model.stats
                            print(model.stats)
                            processed_stats = process_stats(model)
                            processed_stats['evaluation']=evaluate_stats(processed_stats)

                            if model_save:
                                save_model(model, model_name)
                            if stop_criteria(processed_stats):
                                break
                        if stop_criteria(processed_stats):
                            break
                else:
                    print("Test number [of different models . of the same model]: " + str(_test_params) + "." + str(
                        _test_same_model))

                    assert method!=4 or not model_layer_graph
                    model = NN_Residual(get_layers()).to(device) if model_layer_graph else (NN_Advanced(get_layers()).to(device) if method==4 else NN(get_layers()).to(device))
                    if model_load:
                        model = load_model(model_name)

                    # model_copy=None
                    # if calculate_dist:
                    #     model_copy=copy.deepcopy(model).to('cpu')

                    #set_params(model,params)
                    #print("Model parameters: \ngradient_factor: "+str(float(model.gradient_factor))+"   step_factor: "+str(float(model.step_factor))+"    gradient_factor_simple_layers: "+str(float(model.gradient_factor_simple_layers)))
                    print("Params: "+str(params))

                    #if preserve_stats:
                    model.stats = actual_stats
                    model.stats['params']=params
                    set_params(model, params)
                    #optimizer = optim.Adadelta(model.parameters(), lr=lr)
                    # lr = 0.007
                    # if model_nr == 3 or model_nr == 4:
                    #     lr = 0.007
                    # elif model_nr == 5:
                    #     lr = 0.0001
                    # elif model_nr == 6:
                    #     lr = 0.001
                    # optimizer = optim.Adam(model.parameters(), lr)
                    # optimizer = optim.Adam(model.parameters(), lr, betas=(0., 0.999))
                    # if model_nr == 3 or model_nr == 4:
                    #     lr = 0.07
                    # elif model_nr == 5:
                    #     lr = 0.001
                    # elif model_nr == 6:
                    #     lr = 0.01
                    lr=params['lr']
                    #optimizer=optim.RMSprop(model.parameters(), lr)
                    optimizer = optimizer_type(model.parameters(), lr,**optim_args)

                    #optimizer=optim.SGD(model.parameters())
                    #optimizer=optim.SGD(model.parameters(),lr=0.3,momentum=0,nesterov=False)

                    gamma=params['gamma']
                    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

                    # model2=NN_Advanced(copy.deepcopy(model.layers)) if method==4 else NN(copy.deepcopy(model.layers))#model2 = NN(copy.deepcopy(model.layers)).to(device)
                    # model2.train()
                    # model2.load_state_dict(copy.deepcopy(model.state_dict()))
                    model2=copy.deepcopy(model)
                    #opt2 = type(optimizer)(model2.parameters(),lr=0.01,momentum=0,nesterov=False)
                    #opt2 = type(optimizer)(model2.parameters())
                    opt2 = type(optimizer)(model2.parameters(),lr=optimizer.param_groups[0]['lr'],**optim_args)
                    opt2.load_state_dict(copy.deepcopy(optimizer.state_dict()))

                    show_time = True
                    start_time = time.time()
                    for epoch in range(1, epochs + 1):
                        if pretraining is not None and not pretraining['finished']:
                            # global method
                            if pretraining['pretraining_finish_criterion'](actual_stats,epoch-1):
                                pretraining['finished'] = True
                                method = pretraining['method_after_pretraining']
                            else:
                                method = pretraining['method']
                            hyperparameter_index-=1
                            new_params=generate_params()
                            optimizer.param_groups[0]['lr']=new_params['lr']
                            gamma = params['gamma']
                            scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
                            actual_stats['params']=new_params
                            set_params(model,new_params)
                            print("New params: " + str(actual_stats['params']))

                        if mini_batch:
                            train_minibatch(model, device, train_loader, optimizer, epoch, model2, opt2,log_interval=log_interval)
                        else:
                            train(model, device, train_loader, optimizer, epoch, model2, opt2,log_interval=log_interval)
                        if show_time:
                            print("--- %s seconds ---" % (time.time() - start_time))
                        #train(args, model, device, train_loader, optimizer, epoch)#train(args, model, device, train_loader, optimizer, epoch,model2,opt2)
                        test(model, device, test_loader,train_loader,epoch=epoch)
                        scheduler.step()

                        #model.stats['evaluation'] = evaluate_stats(process_stats(model))
                        actual_stats = model.stats
                        print(model.stats)
                        processed_stats=process_stats(model)
                        processed_stats['evaluation'] = evaluate_stats(processed_stats)

                        if model_save:
                            save_model(model,model_name)

                        if stop_criteria(processed_stats):
                            break
                    if stop_criteria(processed_stats):
                        break

                save_state(backup_name,
                           (hyperparameter_index-1, actual_stats, summary, _test_params, _test_same_model + 1,torch.get_rng_state()))

                if force_memory_cleanup and _test_same_model != trainings_same_model - 1:
                    import sys
                    # os.execv(sys.executable, [sys.executable, __file__] + sys.argv)#os.execv(sys.argv[0], sys.argv)
                    os.execv(sys.executable, [sys.executable, __file__] + sys.argv)  #


            if best_stats is None or evaluate_stats(processed_stats)>=evaluate_stats(best_stats):
                best_stats = processed_stats
                print("New best stats:")
                print(best_stats)
                write_to_file(save_name,"New best stats!")
                write_to_file(save_name,str(model.stats))
            write_to_file(save_name, "Actual processed stats:")
            write_to_file(save_name, str(processed_stats))
            write_to_file(save_name, "Best stats so far:")
            write_to_file(save_name, str(best_stats))
            mean,mean_err=plots.utils.mean_and_sem_err(processed_stats['train_loss_min'])
            median, median_err = plots.utils.median_and_sem_err(processed_stats['train_loss_min'])
            additional_stats={'avg_train_loss_min':mean,'pm_avg_train_loss':mean_err,'median_train_loss_min':median,'pm_median_train_loss':median_err}
            if 'train_avg_relative_loss_improvement_avg' in processed_stats:
                additional_stats['train_avg_relative_loss_improvement_avg']=processed_stats['train_avg_relative_loss_improvement_avg']
            summary.append((processed_stats['params'],processed_stats['evaluation'],additional_stats))
            summary=sorted(summary,key=lambda x: x[1])
            write_to_file('summary_'+save_name, str(summary),mode='w')
            write_to_file('short_summary_'+save_name, str([(s[0],s[1]) for s in summary]),mode='w')

            save_state(backup_name,(hyperparameter_index,actual_stats,summary,_test_params+1,0,torch.get_rng_state()))

            if force_memory_cleanup and _test_params != trainings_different_model - 1:
                import sys
                # os.execv(sys.executable, [sys.executable, __file__] + sys.argv)#os.execv(sys.argv[0], sys.argv)
                os.execv(sys.executable, [sys.executable, __file__] + sys.argv)  #

    if __name__ == '__main__':
        main()

    #from imagenet_ood_utils import get_approx_optimal_class_mapping, get_logit_mask, get_dataloaders, load_resnet, convert_resnet_to_module_graph
