import os
import random
import sys
import time
import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from src.efficient_net_loader import CumuloDataset
from src.metrics import scores_per_class
from src.utils import Normalizer, get_dataset_statistics, get_hms, make_directory, get_tile_sampler, tile_collate

# get EfficientNet here
from efficientnet_pytorch import EfficientNet

import torch.nn as nn

from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from src.metrics import scores_per_class

# training hyperparamters
nb_epochs = 200
t_size = 3
nb_classes = 8
batch_size = 1024 # number of tiles per batch 
lr = 0.001
weight_decay = 5e-4

root_dir = os.path.join("/scratch/pvu3/Indep Study/netcdf/month_3/npz")

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic = True

# shape of the input (channels, height, width)
in_shape = (13, t_size, t_size)

use_cuda = torch.cuda.is_available()
print("using GPUs?", use_cuda)

classification_weight = in_shape[0] * in_shape[1] * in_shape[2]

# set up directories to record training, testing and validation result
save_dir = "results/EfficientNet"

save_dir_best = os.path.join(save_dir, "best")
save_dir_last = os.path.join(save_dir, "last")

make_directory(save_dir_best)
make_directory(save_dir_last)

train_log = open(os.path.join(save_dir, "train_log.txt"), 'w')
val_log = open(os.path.join(save_dir, "val_log.txt"), 'w')
test_log = open(os.path.join(save_dir, "test_log.txt"), 'w')

# set up dataset

dataset = CumuloDataset(os.path.join(root_dir, "label/")) # change to unlabelled dir
class_weights, m, s = get_dataset_statistics(dataset, nb_classes, collate=tile_collate, batch_size=40, use_cuda=use_cuda)
print("Length of dataset: {}".format(len(dataset)))


np.save(os.path.join(save_dir, "class-weights.npy"), class_weights)
np.save(os.path.join(save_dir, "mean.npy"), m)
np.save(os.path.join(save_dir, "std.npy"), s)

normalizer = Normalizer(m, s)
class_weights = torch.from_numpy(class_weights).float()

# get train, validation, test sets by randomly splitting the set of swaths
nb_swaths = len(os.listdir(os.path.join(root_dir, "label/")))
idx = np.arange(nb_swaths)
np.random.shuffle(idx)
train_idx, val_idx, test_idx = np.split(idx, [int(.7 * nb_swaths), int(.8 * nb_swaths)])

#train_idx = [0]
#val_idx = [0]
#test_idx = [0]

print("Train id {}  Val id {}    Test id {}".format(train_idx, val_idx, test_idx))


train_dataset = CumuloDataset(os.path.join(root_dir, "label/"), normalizer=normalizer, indices=train_idx) # change to labelled dir
val_dataset = CumuloDataset(os.path.join(root_dir, "label/"), "npz", normalizer=normalizer, indices=val_idx) # change to labelled dir
test_dataset = CumuloDataset(os.path.join(root_dir, "label/"), "npz", normalizer=normalizer, indices=test_idx) # change to labelled dir




# samplers
train_sampler = get_tile_sampler(train_dataset)

# data loaders, batch_size = number of tiles
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1)


# in the following, batch_size corresponds to number of swaths (each swath contain multiple tiles)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, collate_fn=tile_collate, shuffle=False, num_workers=1)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=tile_collate, shuffle=False, num_workers=1)

print("Length trainloader {}".format(len(trainloader)))
print("Length valloader {}".format(len(valloader)))
print("Length testloader {}".format(len(testloader)))


# init model
model = EfficientNet.from_name('efficientnet-b3', num_classes=8)
classes = 13
model._change_in_channels(classes)
#model._conv_stem.in_channels = 13
#a = torch.Tensor(size=(40, 1, 3, 3))
#model._conv_stem.weight = torch.nn.Parameter(torch.cat([model._conv_stem.weight, model._conv_stem.weight, model._conv_stem.weight, model._conv_stem.weight, a], axis=1))




# uses all available GPUs
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
    cudnn.benchmark = True
else:
    in_shapes = model.get_in_shapes()

print('|  Train Epochs: ' + str(nb_epochs))
print('|  Initial Learning Rate: ' + str(lr))

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

best_accuracy, best_f1 = 0., 0.



import torchvision.transforms.functional as F
import torchvision.transforms as transforms

class SquarePad:
	def __call__(self, image):
		w, h = 32, 32
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

# now use it as the replacement of transforms.Pad class
transform=transforms.Compose([
    transforms.ToPILImage(),
    SquarePad(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

def transform_thirteen_channels(inputs):
    new_input = []

    for tile in inputs:
        temp_list = []
        
        for channel in tile:
            transformed_channel = transform(channel)
            temp_list.append(transformed_channel[0])
        
        res = torch.stack(temp_list, dim = 0)
        new_input.append(res)

    new_input = torch.stack(new_input, dim = 0)
    return new_input




def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def train(model, optimizer, epoch, lr, trainloader, train_log, class_weights, use_cuda=False, classification_weight=1, nb_classes=8):
    
    model.train()
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print('|  Number of Trainable Parameters: ' + str(params))
    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, lr))

    conf_matrix = np.zeros((nb_classes, nb_classes))

    if use_cuda:
        class_weights = class_weights.cuda()
    
    superv_criterion = nn.CrossEntropyLoss(weight=class_weights)

    for batch_idx, (_, inputs, *_, labels) in enumerate(trainloader):
        #inputs *= 100
        inputs[inputs != inputs] = 0

        cur_iter = (epoch - 1) * len(trainloader) + batch_idx

        # if first epochs use warmup
        if epoch < 10:
            this_lr = lr * float(cur_iter) / (10 * len(trainloader))
            update_lr(optimizer, this_lr)

        optimizer.zero_grad()

        inputs = transform_thirteen_channels(inputs)
        print(inputs.shape)
        #exit(0)

        if use_cuda:
            inputs = inputs.cuda() # GPU settings
            labels = labels.cuda()
        
        print("Dimension of input {}".format(inputs.shape))

        logits = model(inputs)

        mean_entropy = superv_criterion(logits, labels.long())
        loss = classification_weight * mean_entropy

        # Backward propagation
        loss.backward()

        # Optimizer Update
        optimizer.step()

        _, predicted = torch.max(logits.data, 1)
        conf_matrix += confusion_matrix(labels.data.cpu().numpy(), predicted.cpu().numpy(), labels=np.arange(nb_classes))

        log_dict = {"iter": cur_iter, "loss": loss.item(), "epoch": epoch}
        train_log.write("{}\n".format(json.dumps(log_dict)))
        train_log.flush()

    accuracy_per_class, f1_per_class = scores_per_class(conf_matrix)
    
    # file logging
    log_dict = {"epoch": epoch, "accuracy per class": accuracy_per_class.tolist(), "f1 per class": f1_per_class.tolist()}

    train_log.write("{}\n".format(json.dumps(log_dict)))
    train_log.flush()

    return conf_matrix, np.mean(accuracy_per_class)



def test(model, epoch, testloader, test_log, use_cuda=False, flag="validation", classification_weight=1, nb_classes=8):
    
    model.eval()
    criterion = nn.CrossEntropyLoss()

    nb_tiles = 0.
    test_loss = 0.

    conf_matrix = np.zeros((nb_classes, nb_classes))
    for batch_idx, (inputs, labels) in enumerate(testloader):
        inputs[inputs != inputs] = 0
        nb_tiles += len(labels)

        inputs = transform_thirteen_channels(inputs)

        if use_cuda:
            inputs = inputs.cuda() # GPU settings
            labels = labels.cuda()
        
        logits = model(inputs)

        mean_entropy = criterion(logits, labels.long())
        loss = classification_weight * mean_entropy

        _, predicted = torch.max(logits.data, 1)

        conf_matrix += confusion_matrix(labels.data.cpu().numpy(), predicted.cpu().numpy(), labels=np.arange(nb_classes))

        test_loss += loss.item()

    
    accuracy_per_class, f1_per_class = scores_per_class(conf_matrix)

    # file logging
    log_dict = {"epoch": epoch, "accuracy per class": accuracy_per_class.tolist(), "f1 per class": f1_per_class.tolist()}
    test_log.write("{}\n".format(json.dumps(log_dict)))
    test_log.flush()

    return conf_matrix, np.mean(accuracy_per_class)

def save_model(model, optimizer, train_cm, val_cm, save_dict, save_dir, **kwargs):

    state = save_dict.copy()

    try:
        state['model'] = model.module
        state_dict = model.module.state_dict()

    except AttributeError:
        state['model'] = model
        state_dict = model.state_dict()

    state['model-statedict'] = state_dict
    state['optimizer-statedict'] = optimizer.state_dict()

    for k, value in kwargs.items():
        state[k] = value

    np.save(os.path.join(save_dir, "train-confusion-matrix.npy"), train_cm)
    np.save(os.path.join(save_dir, "val-confusion-matrix.npy"), val_cm)

    torch.save(state, os.path.join(save_dir, 'model.t7'))


elapsed_time = 0

state = {
    'tile-size': "13*3*3",
    'batch-size': batch_size,
    'classifier': 'linear',
    'loss-function': 'cross-entropy',
    'learning-rate': lr,
    'class-weight': classification_weight
}

try:
    
    for epoch in range(1, 1 + nb_epochs):
        start_time = time.time()

        train_cm, train_acc = train(model, optimizer, epoch, lr, trainloader, train_log, class_weights, use_cuda, classification_weight)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))
        print('Training accuracy', train_acc)

        val_cm, val_acc = test(model, epoch, valloader, val_log, use_cuda, "val ", classification_weight)
        print('Validation accuracy', val_acc)
        _, val_f1 = scores_per_class(val_cm)
        val_f1 = val_f1.mean()

        # save state if accuracy or f1 improved
        if val_acc > best_accuracy or val_f1 > best_f1:
            save_model(model, optimizer, train_cm, val_cm, state, save_dir_best, epoch=epoch, train_accuracy=train_acc, val_accuracy=val_acc)
            best_accuracy = val_acc
            best_f1 = val_f1

except KeyboardInterrupt:
    pass
    print("\nWait for the program to save current state")
    print("stopping at epoch", epoch)

except Exception as e:
    raise e

# save last-model
test_cm, _ = test(model, epoch, testloader, test_log, use_cuda, "test ", classification_weight)
save_model(model, optimizer, train_cm, val_cm, state, save_dir_last, epoch=epoch, train_accuracy=train_acc, val_accuracy=val_acc)
np.save(os.path.join(save_dir_last, "test-confusion-matrix.npy"), test_cm)

# test with best-model and save
model = torch.load(os.path.join(save_dir_best, "model.t7"))["model"]

if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
    cudnn.benchmark = True

test_cm, _ = test(model, epoch, testloader, test_log, use_cuda, "test ", classification_weight)
np.save(os.path.join(save_dir_best, "test-confusion-matrix.npy"), test_cm)

# plot confusion matrices
train_cm, val_cm, test_cm = normalize_confusion_matrix(train_cm), normalize_confusion_matrix(val_cm), normalize_confusion_matrix(test_cm)

plt.figure(figsize = (20,5))

plt.subplot(131)

df_cm = pd.DataFrame(train_cm, index = range(8), columns = range(8))

plt.title("TRAIN")
ax = sn.heatmap(df_cm, annot=True, vmin=0, vmax=1)
ax.set(xlabel='predicted', ylabel='target')

plt.subplot(132)

df_cm = pd.DataFrame(val_cm, index = range(8), columns = range(8))

plt.title("VAL")
ax = sn.heatmap(df_cm, annot=True, vmin=0, vmax=1)
ax.set(xlabel='predicted', ylabel='target')

plt.subplot(133)

df_cm = pd.DataFrame(test_cm, index = range(8), columns = range(8))

plt.title("TEST")
ax = sn.heatmap(df_cm, annot=True, vmin=0, vmax=1)
ax.set(xlabel='predicted', ylabel='target')

plt.savefig("EfficientNet-small-confusion-matrices.png", bboxes="tight")




