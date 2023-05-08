import torch
import torchvision.models as models
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook, trange as tnrange
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(42) # try and make the results more reproducible
BASE_PATH = 'MAFood121/'

import os
print(os.listdir("MAFood121/images"))
epochs = 10
batch_size = 32
SMALL_DATA = False
IMG_SIZE = (384, 384)

import pandas as pd

train_df = pd.read_hdf("trainSL_df.h5")
val_df = pd.read_hdf("testSL_df.h5")
test_df = pd.read_hdf("testSL_df.h5")

if SMALL_DATA:
    train_df = train_df[:128]
    val_df = test_df[:128]
    test_df = test_df[:128]

col_names = list(train_df.columns.values)

ing_names = col_names[:-3]
targets = ing_names

import torch.utils.data as data

class DataWrapper(data.Dataset):
    ''' Data wrapper for pytorch's data loader function '''
    def __init__(self, image_df, resize):
        self.dataset = image_df
        self.resize = resize

    def __getitem__(self, index):
        c_row = self.dataset.iloc[index]
        target_arr = []
        for item in c_row[targets].values:
            target_arr.append(item)

        image_path, target = c_row['path'], torch.from_numpy(np.array(target_arr)).float()  #image and target
        #read as rgb image, resize and convert to range 0 to 1
        image = cv2.imread(image_path, 1)
        if self.resize:
            image = cv2.resize(image, IMG_SIZE)/255.0 
        else:
            image = image/255.0
        image = (torch.from_numpy(image.transpose(2,0,1))).float() #NxCxHxW
        return image, target

    def __len__(self):
        return self.dataset.shape[0]  
    
from torchvision.models import ResNet50_Weights
import torch.nn as nn

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(targets))

ct = 0
for name, child in model.named_children():
    ct += 1
    if ct < 8:
        for name2, params in child.named_parameters():
            params.requires_grad = False
            
import torch.optim as optim

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_dataset = DataWrapper(train_df, True)
train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True, batch_size=batch_size, pin_memory=False)

val_dataset = DataWrapper(val_df, True)
val_loader = torch.utils.data.DataLoader(val_dataset,shuffle=True, batch_size=batch_size, pin_memory=False)

test_dataset = DataWrapper(test_df, True)
test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=True, batch_size=batch_size, pin_memory=False)

#TRY LATER
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):

    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

from collections import defaultdict
import matplotlib.pyplot as plt

train_results = defaultdict(list)
train_iter, test_iter, best_acc = 0,0,0
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 10))
ax1.set_title('Train Loss')
ax2.set_title('Train Accuracy')
ax3.set_title('Test Loss')
ax4.set_title('Test Accuracy')

f1_scores = defaultdict(list)

import numpy as np
import cv2

for i in tqdm(range(epochs), desc='Epochs'):
    print("Epoch ",i)
    # Model switches to train phase
    model.train() 
    all_outputs = []
    all_targets = []
    import torch.nn.functional as F

    count, loss_val, correct, total = train_iter, 0, 0, 0
    with tqdm(train_loader, desc='Training', total=len(train_loader), miniters=1) as pbar:
        for img_data, target in pbar:
            img_data, target = img_data.to(device), target.to(device)
            output = model(img_data) #FWD prop

            loss = criterion(output, target) #Cross entropy loss
            c_loss = loss.data.item()
            ax1.plot(count, c_loss, 'r.')
            loss_val += c_loss

            optimizer.zero_grad() #Zero out any cached gradients
            loss.backward() #Backward pass
            optimizer.step() #Update the weights

            total_batch = (target.size(0) * target.size(1))
            total += total_batch
            output_data = F.softmax(output, dim=1)
            output_data = torch.argmax(output_data, dim=1)
            target_data = torch.argmax(target, dim=1)
            for arr1,arr2 in zip(output_data, target_data):
                all_outputs.append(arr1.cpu().numpy())
                all_targets.append(arr2.cpu().numpy())
            c_acc = torch.sum((output_data == target_data.to(device)).to(torch.float)).item()
            ax2.plot(count, c_acc/total_batch, 'r.')
            correct += c_acc
            count +=1
                
    from sklearn.metrics import f1_score, recall_score, precision_score

    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    f1score_samples = f1_score(y_true=all_targets, y_pred=all_outputs, average='macro')
    f1score_macro = f1_score(y_true=all_targets, y_pred=all_outputs, average='macro')
    f1score_weighted = f1_score(y_true=all_targets, y_pred=all_outputs, average='weighted')
    recall = recall_score(y_true=all_targets, y_pred=all_outputs, average='macro', zero_division=1)
    prec = precision_score(y_true=all_targets, y_pred=all_outputs, average='macro', zero_division=1)
    hamming = hamming_score(y_true=all_targets, y_pred=all_outputs)

    f1_scores["samples_train"].append(f1score_samples)
    f1_scores["macro_train"].append(f1score_macro)
    f1_scores["weighted_train"].append(f1score_weighted)
    f1_scores["hamming_train"].append(hamming)

    train_loss_val, train_iter, train_acc = loss_val/len(train_loader.dataset), count, correct/float(total)

    print("Training loss: ", train_loss_val, " train acc: ",train_acc)

    #Model switches to test phase
    model.eval()  
    all_outputs = []
    all_targets = []
    
    count, correct, total, lost_val = test_iter, 0, 0, 0
    for img_data, target in tqdm(val_loader, desc='Testing', total=len(val_loader)):
        img_data, target = img_data.to(device), target.to(device)
        output = model(img_data)
        loss = criterion(output, target) #Cross entropy loss
        c_loss = loss.data.item()
        ax3.plot(count, c_loss, 'b.')
        loss_val += c_loss
        total_batch = (target.size(0) * target.size(1))
        total += total_batch
        output_data = torch.sigmoid(output)>=0.5
        target_data = (target==1.0)
        for arr1,arr2 in zip(output_data, target_data):
            all_outputs.append(list(arr1.cpu().numpy()))
            all_targets.append(list(arr2.cpu().numpy()))
        c_acc = torch.sum((output_data == target_data.to(device)).to(torch.float)).item()
        ax4.plot(count, c_acc/total_batch, 'b.')
        correct += c_acc
        count += 1
        
    #F1 Score
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    print(all_outputs)
    print(all_targets)
    f1score_samples = f1_score(y_true=all_targets, y_pred=all_outputs, average='macro')
    f1score_macro = f1_score(y_true=all_targets, y_pred=all_outputs, average='macro')
    f1score_weighted = f1_score(y_true=all_targets, y_pred=all_outputs, average='weighted')
    hamming = hamming_score(y_true=all_targets, y_pred=all_outputs)
    recall = recall_score(y_true=all_targets, y_pred=all_outputs, average='macro', zero_division=1)
    prec = precision_score(y_true=all_targets, y_pred=all_outputs, average='macro', zero_division=1)

    f1_scores["samples_test"].append(f1score_samples)
    f1_scores["macro_test"].append(f1score_macro)
    f1_scores["weighted_test"].append(f1score_weighted)
    f1_scores["hamming_test"].append(hamming)
    
    #Accuracy over entire dataset
    test_acc, test_iter, test_loss_val = correct/float(total), count, loss_val/len(test_loader.dataset)
    print("Test set accuracy: ",test_acc)
    
    train_results['epoch'].append(i)
    train_results['train_loss'].append(train_loss_val)
    train_results['train_acc'].append(train_acc)
    train_results['train_iter'].append(train_iter)
    
    train_results['test_loss'].append(test_loss_val)
    train_results['test_acc'].append(test_acc)
    train_results['test_iter'].append(test_iter)
    
    #Save model with best accuracy
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth') 
fig.savefig('train_curves.png')
        
print("TRAIN")
print("F1 Samples: ", f1_scores["samples_train"])
print("F1 Weighted: ", f1_scores["weighted_train"])
print("Hamming: ", f1_scores["hamming_train"])
print()
print("==============")
print("VALIDATION")
print("F1 Samples: ", f1_scores["samples_test"])
print("F1 Weighted: ", f1_scores["weighted_test"])
print("Hamming: ", f1_scores["hamming_test"])

#Inference on test
model_path = "best_model.pth"
model.load_state_dict(torch.load(model_path))
model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()

#Run predictions
all_outputs = []
all_targets = []
for img_data, target in tqdm_notebook(test_loader, desc='Testing'):
    img_data, target = img_data.to(device), target.to(device)
    output = model(img_data)
    loss = criterion(output, target) #Cross entropy loss
    c_loss = loss.data.item()
    ax3.plot(count, c_loss, 'b.')
    loss_val += c_loss
    total_batch = (target.size(0) * target.size(1))
    total += total_batch
    output_data = torch.sigmoid(output)>=0.5
    target_data = (target==1.0)
    for arr1,arr2 in zip(output_data, target_data):
        all_outputs.append(list(arr1.cpu().numpy()))
        all_targets.append(list(arr2.cpu().numpy()))
    c_acc = torch.sum((output_data == target_data.to(device)).to(torch.float)).item()
    ax4.plot(count, c_acc/total_batch, 'b.')
    correct += c_acc
    count += 1
    
#F1 Score
all_outputs = np.array(all_outputs)
all_targets = np.array(all_targets)
f1score_samples = f1_score(y_true=all_targets, y_pred=all_outputs, average='macro')
f1score_macro = f1_score(y_true=all_targets, y_pred=all_outputs, average='macro')
f1score_weighted = f1_score(y_true=all_targets, y_pred=all_outputs, average='weighted')
recall = recall_score(y_true=all_targets, y_pred=all_outputs, average='macro', zero_division=1)
prec = precision_score(y_true=all_targets, y_pred=all_outputs, average='macro')
hamming = hamming_score(y_true=all_targets, y_pred=all_outputs)

print("TEST")
print("F1 Samples: ", f1score_samples)
print("F1 Weighted: ", f1score_weighted)
print("Hamming: ", hamming)