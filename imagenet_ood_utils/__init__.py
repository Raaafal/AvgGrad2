import copy
import os
import shutil

import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
#import main_skip_connection_support
# from main_skip_connection_support import DependencyGraphNode,DependencyGraph,NN_Residual,average_gradient_of_nonlinear_layers_enhancement
#from main_skip_connection_support import DependencyGraphNode,DependencyGraph,NN_Residual#,average_gradient_of_nonlinear_layers_enhancement
from algorithms import *
#from main_skip_connection_support import *
import torch.nn.functional as F
import gc
import psutil
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"


#average_gradient_of_nonlinear_layers_enhancement=False
additional_prefix=''
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#device='cpu'

def structure_files():
    path=additional_prefix+'../data/imagenetood/'#'../data/imagenet-mini/'
    train_images = os.listdir(path)
    if not os.path.exists(path + 'train'):
        os.makedirs(path + 'train', exist_ok=True)
    for img in train_images:
        if os.path.isfile(path+img):
            src = path + img
            destination = path + 'train/' + img
            print('Moving '+src+' to '+destination)
            shutil.move(src, destination)


    train_images = os.listdir(path+'train')
    for image in train_images:
        if os.path.isfile(path+'train/'+image):
            # split = image.split('_')
            # cls_name = split[0]
            cls_name=image[0:9]#image[0:10]

            if not os.path.exists(path+'train/' + cls_name):
                # print('creating dir: ', 'train/' + cls_name)
                os.makedirs(path+'train/' + cls_name, exist_ok=True)

            src = path+'train/' + image
            destination = path+'train/' + cls_name + '/'
            # print('moving')
            # print(src)
            # print(destination)
            print('Moving ' + src + ' to ' + destination)
            shutil.move(src, destination)
def training_set_stats(set='train'):
    stats={}
    path = additional_prefix + '../data/imagenetood/'
    classes=os.listdir(path+set+'/')
    for cls in classes:
        num_examples=len(os.listdir(path+set+'/'+cls))
        if num_examples in stats:
            stats[num_examples]+=1
        else:
            stats[num_examples]=1
    for num_examples,count in stats.items():
        print(str(num_examples)+' examples were for '+str(count)+' classes')
def stratified_sampling_test_split():
    #return
    np.random.seed(123)

    path = additional_prefix + '../data/imagenetood/'#'../data/imagenet-mini/'
    if not os.path.isdir(path + '/test'):
        os.makedirs(path + '/test', exist_ok=True)
    if len(os.listdir(path + '/test'))>0:
        print('Test file already not empty, sampling stopped')
        return

    train_classes = os.listdir(path + '/train')
    for train_class in train_classes:
        if os.path.isdir(path + '/train/'+train_class):
            imgs=os.listdir(path + '/train/'+train_class)
            count = len(imgs)
            if count>=47 and count<=50:
                while count-len(imgs)<10:
                    randind=np.random.randint(len(imgs))
                    src = path + 'train/' + train_class+'/'+imgs[randind]
                    destination = path + 'test/' + train_class + '/'
                    print('Moving '+src+' to '+destination)
                    if not os.path.isdir(path + 'test/' + train_class + '/'):
                        os.makedirs(path + 'test/' + train_class + '/', exist_ok=True)
                    #imgs.pop(randind)
                    shutil.move(src, destination)
                    imgs = os.listdir(path + '/train/' + train_class)


def get_dataloaders(batch_size=128):
    # Data loading code
    # if args.dummy:
    #     print("=> Dummy data is used!")
    #     train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
    #     val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    # else:
    data_path=additional_prefix+'../data/imagenetood'#'../data/imagenet-mini'
    #traindir = os.path.join(data_path, 'train')
    traindir=data_path+'/train'
    #valdir = os.path.join(data_path, 'val')
    testdir = data_path + '/test'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    # else:
    #     train_sampler = None
    #     val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=torch.get_num_threads()-1, pin_memory=True, sampler=None)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=torch.get_num_threads()-1, pin_memory=True, sampler=None)
    print('Number of threads for loading data: '+str(torch.get_num_threads()-1))
    return train_loader,test_loader

def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output

def load_resnet():
    #model from https: // github.com / facebookarchive / fb.resnet.torch / blob / master / pretrained / README.md
    # from torch.utils.serialization import load_lua
    # x = load_lua('x.t7')
    #resnet = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
    resnet = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
    return resnet

def read_file(file_name):
    try:
        with open(file_name, 'r') as f:
            s = f.read()
            #return ast.literal_eval(s)
            return eval(s)
    except Exception as e:  # works on python 3.x
        print(repr(e))
    return None
# def get_logit_mask():
#     mask=torch.ones((1000,))*-torch.inf
#     class_names_to_idx=cls_to_idx.get_cls_to_idx()#read_file()
#     #for (cls,idx) in class_names_to_idx.items():
#     data_path = additional_prefix + '../data/imagenetood'  # '../data/imagenet-mini'
#     # traindir = os.path.join(data_path, 'train')
#     traindir = data_path + '/train'
#     classes=os.listdir(traindir)
#     matched=0
#     not_matched=0
#     for cls in classes:
#         if cls in class_names_to_idx:
#             matched+=1
#             mask[class_names_to_idx[cls]] = 0.
#         else:
#             not_matched+=1
#     print('Classes that are in both imagenet and imagenet_ood: '+str(matched))
#     print('Classes that are in imagenet_ood, but not in imagenet: '+str(not_matched))
#     return mask
def get_logit_mask(mapping):
    mask = torch.ones((1000,)) * -torch.inf
    for ind in mapping:
        mask[ind]=0.
    return mask

def save_state(file_name,data):
    #write_to_file(file_name,data)
    torch.set_printoptions(profile="full")#otherwise "default"
    tmp_name=file_name[:file_name.rfind('.')]+'_tmp'+file_name[file_name.rfind('.'):]
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

def get_approx_optimal_class_mapping(train_loader):
    mapping_file_name=additional_prefix+'../data/imagenet_ood_class_mapping.txt'

    mapping=read_state(mapping_file_name)

    if mapping is None:
        model = load_resnet()
        model=model.to(device)

        class_confusion_matrix=torch.zeros((1000,637),dtype=torch.int32)

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                out = model(data)
                out=torch.argmax(out,1)
                for i in range(out.shape[0]):
                    class_confusion_matrix[out[i]][target[i]]+=1
                print(out)
                print(batch_idx)
        print(class_confusion_matrix)
        class_confusion_matrix_copy=class_confusion_matrix.clone().detach()

        prediction_count=torch.sum(class_confusion_matrix)
        ood_to_imagenet_class_mapping={}
        while len(ood_to_imagenet_class_mapping.items())<637:
            for i in range(1000):
                ood_class_candidate=int(torch.argmax(class_confusion_matrix[i]))
                classification_num=int(class_confusion_matrix[i][ood_class_candidate])
                if ood_class_candidate in ood_to_imagenet_class_mapping:
                    if ood_to_imagenet_class_mapping[ood_class_candidate]==i:
                        continue
                    #print('Mapping clash!')
                    if (class_confusion_matrix[ood_to_imagenet_class_mapping[ood_class_candidate],ood_class_candidate]<classification_num
                        and i>ood_to_imagenet_class_mapping[ood_class_candidate])\
                            or (class_confusion_matrix[ood_to_imagenet_class_mapping[ood_class_candidate],ood_class_candidate]<classification_num
                                and i<ood_to_imagenet_class_mapping[ood_class_candidate]):
                        ood_to_imagenet_class_mapping[ood_class_candidate]=i
                    else:
                        class_confusion_matrix[i][ood_class_candidate]-=1
                else:
                    ood_to_imagenet_class_mapping[ood_class_candidate] = i

            print('Progress: '+str(len(ood_to_imagenet_class_mapping.items()))+'/'+str(637),end='\r')

        correct_predictions=0
        for i in range(637):
            correct_predictions+=class_confusion_matrix_copy[ood_to_imagenet_class_mapping[i],i]

        print('Accuracy: '+str(100.0*correct_predictions/prediction_count))

        ood_to_imagenet_class_mapping_array=torch.ones((637,),dtype=torch.int32)*-1
        for i in range(637):
            ood_to_imagenet_class_mapping_array[i]=ood_to_imagenet_class_mapping[i]

        mapping=ood_to_imagenet_class_mapping_array
        save_state(mapping_file_name,mapping)
    return mapping

def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if not children:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            output[name] = nested_children(child)
    return output

from collections.abc import Iterable
def convert_resnet_to_module_graph(resnet):
    #prev_long_connection=None
    actual_node=None

    modules=nested_children(resnet)
    for (name,component) in modules.items():
        if isinstance(component, Iterable):
            for (name2, component2) in component.items():
                if isinstance(component2, Iterable):
                    module_list=list(component2.items())
                    prev_long_connection=actual_node
                    for i in range(len(module_list)):
                        (name3, component3)=module_list[i]
                        if i==2 or i==4:#a patch
                            actual_node = DependencyGraphNode(torch.nn.ReLU(inplace=False),
                                                                                           [actual_node])
                            actual_node.name='ReLU (additional)'

                        if i+2==len(module_list) and module_list[i+1][0]=='downsample':#i+1<len(module_list):
                            #if module_list[i+1][0]=='downsample':
                            assert i+2==len(module_list)
                            connect_to_relu=actual_node
                            first_bottleneck_node=True
                            for (name4,component4) in module_list[i+1][1].items():
                                actual_node = DependencyGraphNode(component4,
                                                                                               [None if first_bottleneck_node else actual_node,
                                                                                                prev_long_connection])
                                actual_node.name=name+'_'+name2+'_'+module_list[i+1][0]+'_'+name4
                                prev_long_connection=None
                                first_bottleneck_node=False
                            actual_node = DependencyGraphNode(component3,
                                                                                           [actual_node,
                                                                                               connect_to_relu])
                            actual_node.name = name + '_' + name2 + '_' + name3
                            break
                        elif i+1==len(module_list):
                            assert name3=='relu'
                            # actual_node = DependencyGraphNode(component3,
                            #                                                                [actual_node,
                            #                                                                 prev_long_connection])
                            actual_node = DependencyGraphNode(torch.nn.ReLU(inplace=False),
                                                                                           [actual_node,
                                                                                            prev_long_connection])
                            actual_node.name = name + '_' + name2 + '_' + name3
                            break
                        else:
                            actual_node = DependencyGraphNode(component3,
                                                                                           [actual_node])
                            actual_node.name = name + '_' + name2 + '_' + name3

                else:
                    actual_node = DependencyGraphNode(component2,
                                                                                   [actual_node])
                    actual_node.name = name + '_' + name2
        else:
            if isinstance(component,torch.nn.Linear):
                actual_node=DependencyGraphNode(torch.nn.Flatten(start_dim=1),[actual_node])
                #actual_node=DependencyGraphNode(lambda x:torch.flatten(x,1),[actual_node])
                actual_node.name='flatten'
            if isinstance(component,torch.nn.ReLU):
                actual_node = DependencyGraphNode(torch.nn.ReLU(inplace=False), [actual_node])
                actual_node.name = 'ReLU'
            else:
                actual_node=DependencyGraphNode(component,[actual_node])
                actual_node.name = name

    graph=DependencyGraph(actual_node)
    for node in graph.nodes_ordered:
        if type(node.module)==torch.nn.ReLU and node.module.inplace:
            node.module.inplace=False
    return graph
def resnet_to_model_supporting_avg_grad(resnet):
    dependency_graph=convert_resnet_to_module_graph(resnet)
    model=NN_Residual(dependency_graph)
    return model
def switch_RELU_to_GELU_in_dependency_graph(dependency_graph):
    for node in dependency_graph.nodes_ordered:
        if isinstance(node.module,torch.nn.ReLU):
            node.module=torch.nn.GELU()
            node.name=node.module.__class__.__name__
    return dependency_graph
def switch_RELU_to_SiLU_in_dependency_graph(dependency_graph):
    for node in dependency_graph.nodes_ordered:
        if isinstance(node.module,torch.nn.ReLU):
            node.module=torch.nn.SiLU()
            node.name=node.module.__class__.__name__
    return dependency_graph

def switch_activations_in_dependency_graph(dependency_graph,activation_after,activation_before=torch.nn.ReLU):
    for node in dependency_graph.nodes_ordered:
        if isinstance(node.module,activation_before):
            node.module=activation_after()
            node.name=node.module.__class__.__name__
    return dependency_graph
def load_resnet_with_GELU():
    resnet=load_resnet()
    resnet=resnet_to_model_supporting_avg_grad(resnet)
    for node in resnet.dependencyGraph.nodes_ordered:
        if isinstance(node.module,torch.nn.ReLU):
            node.module=torch.nn.GELU()
    resnet_GELU=NN_Residual(resnet.dependencyGraph)
    return resnet_GELU
def learning_test():
    global additional_prefix
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    additional_prefix = '../'
    structure_files()
    stratified_sampling_test_split()
    print('Training set:')
    training_set_stats(set='train')
    print('Test set:')
    training_set_stats(set='test')
    # train_loader,test_loader=get_dataloaders(batch_size=64)
    train_loader, test_loader = get_dataloaders(batch_size=32)

    mapping = get_approx_optimal_class_mapping(train_loader)
    logit_mask = get_logit_mask(mapping)
    logit_mask = logit_mask.to(device)
    model = load_resnet()
    model = model.to(device)
    mapping = mapping.to(device).long()
    mapping.requires_grad = False
    logit_mask = logit_mask.to(device)

    correct = 0
    predicted = 0

    # optimizer=torch.optim.RMSprop(model.parameters(),lr=0.)#1e-4)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-5)

    log_interval = 1

    model.train()
    # model.eval()
    # with torch.no_grad():
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        out = model(data)

        loss = torch.nn.functional.cross_entropy(out + logit_mask, mapping[target])
        # loss.requires_grad = True

        loss.backward()
        optimizer.step()

        out = torch.argmax(out + logit_mask, 1)

        print(out)
        print(batch_idx)

        # correct+=torch.sum(mapping[target] == out)
        correct += torch.count_nonzero(mapping[target] == out, 0)
        predicted += target.shape[0]

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                1, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))

    print('Accuracy: ' + str(100.0 * correct / predicted))

def inference_test():
    global additional_prefix
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    additional_prefix = '../'
    structure_files()
    stratified_sampling_test_split()
    print('Training set:')
    training_set_stats(set='train')
    print('Test set:')
    training_set_stats(set='test')
    # train_loader,test_loader=get_dataloaders(batch_size=64)
    train_loader, test_loader = get_dataloaders(batch_size=6)

    mapping = get_approx_optimal_class_mapping(train_loader)
    logit_mask = get_logit_mask(mapping)
    logit_mask = logit_mask.to(device)
    model = load_resnet()
    model = model.to(device)
    mapping = mapping.to(device).long()
    mapping.requires_grad = False
    logit_mask = logit_mask.to(device)

    correct = 0
    predicted = 0

    # optimizer=torch.optim.RMSprop(model.parameters(),lr=0.)#1e-4)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

    log_interval = 1

    model.eval()
    # model.eval()
    # with torch.no_grad():
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #optimizer.zero_grad()

        out = model(data)

        loss = torch.nn.functional.cross_entropy(out + logit_mask, mapping[target])
        ## loss.requires_grad = True

        #loss.backward()
        #optimizer.step()

        out = torch.argmax(out + logit_mask, 1)

        print(out)
        print(batch_idx)

        # correct+=torch.sum(mapping[target] == out)
        correct += torch.count_nonzero(mapping[target] == out, 0)
        predicted += target.shape[0]

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                1, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))

    print('Accuracy: ' + str(100.0 * correct / predicted))

def inference_test_custom_graph_class():
    global additional_prefix
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    additional_prefix = '../'
    structure_files()
    stratified_sampling_test_split()
    print('Training set:')
    training_set_stats(set='train')
    print('Test set:')
    training_set_stats(set='test')
    # train_loader,test_loader=get_dataloaders(batch_size=64)
    train_loader, test_loader = get_dataloaders(batch_size=16)

    mapping = get_approx_optimal_class_mapping(train_loader)
    logit_mask = get_logit_mask(mapping)
    logit_mask = logit_mask.to(device)
    resnet = load_resnet()

    model=resnet_to_model_supporting_avg_grad(resnet)
    model.dependencyGraph.show_graph()

    model = model.to(device)
    mapping = mapping.to(device).long()
    mapping.requires_grad = False
    logit_mask = logit_mask.to(device)

    correct = 0
    predicted = 0

    # optimizer=torch.optim.RMSprop(model.parameters(),lr=0.)#1e-4)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

    log_interval = 1

    model.eval()
    # model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            #optimizer.zero_grad()

            out = model(data)

            loss = torch.nn.functional.cross_entropy(out + logit_mask, mapping[target])
            ## loss.requires_grad = True

            #loss.backward()
            #optimizer.step()

            out = torch.argmax(out + logit_mask, 1)

            print(out)
            print(batch_idx)

            # correct+=torch.sum(mapping[target] == out)
            correct += torch.count_nonzero(mapping[target] == out, 0)
            predicted += target.shape[0]

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    1, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        print('Accuracy: ' + str(100.0 * correct / predicted))

def average_gradient_training_test():
    global additional_prefix
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    additional_prefix = '../'
    structure_files()
    stratified_sampling_test_split()
    print('Training set:')
    training_set_stats(set='train')
    print('Test set:')
    training_set_stats(set='test')
    # train_loader,test_loader=get_dataloaders(batch_size=64)
    train_loader, test_loader = get_dataloaders(batch_size=6)

    mapping = get_approx_optimal_class_mapping(train_loader)
    logit_mask = get_logit_mask(mapping)
    logit_mask = logit_mask.to(device)
    torch.manual_seed(1)
    resnet = load_resnet()

    torch.manual_seed(1)
    resnet2=load_resnet()

    model = resnet_to_model_supporting_avg_grad(resnet)
    model.dependencyGraph.show_graph()

    model2=resnet_to_model_supporting_avg_grad(resnet2)
    model2.dependencyGraph.show_graph()
    model2=copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())



    model = model.to(device)
    model2=model2.to(device)

    mapping = mapping.to(device).long()
    mapping.requires_grad = False
    logit_mask = logit_mask.to(device)

    correct = 0
    predicted = 0

    # optimizer=torch.optim.RMSprop(model.parameters(),lr=0.)#1e-4)
    lr=1e-4
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    log_interval = 1


    #import sys
    #sys.setrecursionlimit(20000)
    #model2=copy.deepcopy(model)
    #import os
    #save_model(open(os.path.abspath('model.txt'),'w+'),model)
    #model2=load_resnet(open(os.path.abspath('model.txt')))
    #model2=NN_Residual([None])
    #model2.load_state_dict(model.state_dict())
    #resnet2=load_resnet()
    # resnet2= torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
    # model2 = resnet_to_model_supporting_avg_grad(resnet)
    # model2.load_state_dict(model.state_dict())
    # resnet2=copy.deepcopy(resnet)#torchvision.models.resnet152(weights=None)
    # model2=resnet_to_model_supporting_avg_grad(resnet2)
    # model2.load_state_dict(copy.deepcopy(model.state_dict()))
    #model2.dependencyGraph=copy.deepcopy(model.dependencyGraph)
    #del resnet
    #del resnet2
    model2.dependencyGraph.show_graph()#copy.deepcopy(model)
    #opt2=copy.deepcopy(optimizer)
    opt2=type(optimizer)(model2.parameters(),lr=optimizer.param_groups[0]['lr'])
    model2.train()
    # model.eval()
    model.train()

    total_relative_loss_improvement_denominator = 0.
    total_loss_improvement = 0.
    relative_loss_improvement_denominator = 0.
    loss_improvement = 0.
    higher_loss_batch_counter = 0
    lower_loss_batch_counter = 0
    batch_counter = 0
    with_optimizer_parameter_copy = False
    epoch=1

    #logit_mask.requires_grad=True

    method=2#1 or 2

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target=mapping[target]
        # optimizer.zero_grad()

        # out = model(data)
        #
        # loss = torch.nn.functional.cross_entropy(out + logit_mask, mapping[target])
        # ## loss.requires_grad = True
        #
        # # loss.backward()
        # # optimizer.step()
        #
        # out = torch.argmax(out + logit_mask, 1)
        #
        # print(out)
        # print(batch_idx)
        #
        # # correct+=torch.sum(mapping[target] == out)
        # correct += torch.count_nonzero(mapping[target] == out, 0)
        # predicted += target.shape[0]
        #
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         1, batch_idx * len(data), len(train_loader.dataset),
        #            100. * batch_idx / len(train_loader), loss.item()))

        iter_count = 1
        for i in range(iter_count):
            model2.load_state_dict(model.state_dict())
            # x=optimizer.state_dict()
            if with_optimizer_parameter_copy:
                opt2.load_state_dict(optimizer.state_dict())

            model2.set_require_grad(False)
            #torch.cuda.empty_cache()
            print('5. '+str(torch.cuda.memory_allocated()))
            output = model2(data)
            print('6. '+str(torch.cuda.memory_allocated()))
            output=output+logit_mask
            loss = F.cross_entropy(output, target)
            #loss.requires_grad = True
            loss_initial=loss.item()

            with torch.no_grad():
                correct += torch.count_nonzero(target == torch.argmax(output, 1), 0)
                predicted += target.shape[0]

            #model2.set_require_grad(average_gradient_of_nonlinear_layers_enhancement)

            if i == 0:
                opt2.zero_grad()
                loss.backward(retain_graph=False)

            else:
                model2.copy_grad_from(model)

            print('7. '+str(torch.cuda.memory_allocated()))
            opt2.step()

            del loss
            del output

            # if average_gradient_of_nonlinear_layers_enhancement:
            #     model2.delete_all_inputs_and_outputs_that_are_not_nonlinear()

            output1 = model(data)
            #output1+=logit_mask
            loss1 = torch.nn.functional.cross_entropy(output1 + logit_mask, target)  # F.nll_loss(output1, target)
            #loss1.requires_grad=True

            model2.set_require_grad(True)
            output = model2(data)
            #output += logit_mask
            loss = torch.nn.functional.cross_entropy(output + logit_mask, target)  # F.nll_loss(output, target)#todo check what this line changes
            #loss.requires_grad = True

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

            print('1. '+str(torch.cuda.memory_allocated()))
            if method==2:
                model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=False,
                                                                   accumulate_gradients=False, gradient_only_modification=True)
                #todo delete all not necessary grad tensors in backward_grad_correction_with_weight_change2

                print('2. '+str(torch.cuda.memory_allocated()))
                optimizer.step()
            elif method==1:
                model.backward_grad_correction_with_weight_change2(loss1, model2, weight_change=True,
                                                                   accumulate_gradients=False,
                                                                   gradient_only_modification=False)
            print('3. '+str(torch.cuda.memory_allocated()))

            loss_to_del=loss
            loss1_to_del=loss1
            loss=loss.item()
            loss1=loss1.item()
            del loss_to_del,loss1_to_del
            del output,output1
            model.input_output_cleanup()
            print('3.2. '+str(torch.cuda.memory_allocated()))

            loss2=None
            with torch.no_grad():
                output2 = model(data)
                loss2 = F.cross_entropy(output2+logit_mask, target)  # F.nll_loss(output2, target)

                print('4. '+str(torch.cuda.memory_allocated()))

            batch_counter += 1
            if loss < loss2:
                higher_loss_batch_counter += 1
                high_loss = True
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
                        higher_loss_batch_counter / batch_counter) + "   gradient_factor: " + str(
                        model.gradient_factor) + "   step factor:" + str(
                        model.step_factor) + " Lower loss batch ratio: " + str(
                        lower_loss_batch_counter / batch_counter) + " Avg loss improvement: " + str(
                        loss_improvement / batch_counter) + " Avg relative loss improvement: " + str(
                        (loss_improvement / abs(
                            relative_loss_improvement_denominator)) if relative_loss_improvement_denominator!=0 else "Division by zero") + " Total avg relative loss improvement: " + str(
                        total_loss_improvement / abs(total_relative_loss_improvement_denominator) if total_relative_loss_improvement_denominator!=0 else "Division by zero"))

            #cleanup
            # del loss,loss1,loss2
            # del output,output1,output2
            # model.input_output_cleanup()
            #model2.input_output_cleanup()

            del output2,loss2
            model.input_output_cleanup()
            print('4.2. '+str(torch.cuda.memory_allocated()))

    print('Accuracy: ' + str(100.0 * correct / predicted))


# average_gradient_training_test()
if __name__=='__main__':
    #learning_test()
    #inference_test()
    #inference_test_custom_graph_class()
    average_gradient_training_test()
else:
    if os.path.exists(path=additional_prefix+'../data/imagenetood/'):
        structure_files()
        stratified_sampling_test_split()

#from main_skip_connection_support import DependencyGraphNode,DependencyGraph,NN_Residual,average_gradient_of_nonlinear_layers_enhancement
