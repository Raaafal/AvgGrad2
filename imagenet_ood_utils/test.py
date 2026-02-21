import os
import shutil

import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import cls_to_idx

additional_prefix=''
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

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
        num_workers=4, pin_memory=True, sampler=None)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, sampler=None)
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

def load_model():
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
        model = load_model()
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



if __name__=='__main__':
    #device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    additional_prefix='../'
    structure_files()
    stratified_sampling_test_split()
    print('Training set:')
    training_set_stats(set='train')
    print('Test set:')
    training_set_stats(set='test')
    #train_loader,test_loader=get_dataloaders(batch_size=64)
    train_loader,test_loader=get_dataloaders(batch_size=128)

    mapping=get_approx_optimal_class_mapping(train_loader)
    logit_mask=get_logit_mask(mapping)
    logit_mask=logit_mask.to(device)
    model=load_model()
    model=model.to(device)
    mapping=mapping.to(device).long()
    logit_mask=logit_mask.to(device)

    correct=0
    predicted=0

    optimizer=torch.optim.RMSprop(model.parameters(),lr=1e-4)

    log_interval=1

    #model.eval()
    with torch.no_grad():
        for batch_idx, (data,target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            out=model(data)

            loss = torch.nn.functional.cross_entropy(out+logit_mask, mapping[target])
            loss.requires_grad = True
            loss.backward()
            optimizer.step()

            out = torch.argmax(out+logit_mask, 1)

            print(out)
            print(batch_idx)

            #correct+=torch.sum(mapping[target] == out)
            correct+=torch.count_nonzero(mapping[target] == out,0)
            predicted+=target.shape[0]

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print('Accuracy: '+str(100.0*correct/predicted))
