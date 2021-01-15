
# coding: utf-8

# ## If you are looking for a <span style="color:GOLD">team member, do consider me too !

# 
# References :
# 1. @abhishek and @artgor 's Parallel Programming video https://www.youtube.com/watch?v=VRVit0-0AXE
# 2. @yasufuminakama 's Amazying Notebook https://www.kaggle.com/yasufuminakama/moa-pytorch-nn-starter 
# 
# # `If you consider forking my kernel, remember to turn off the internet after giving an` **<span style="color:GREEN">UPVOTE**

# ## Update V.14
# 1. Added features from PCA to existing ones [improves CV score]
# 
# ## Update V.11
# 1. Added feature selection using `VarianceThreshold` method of sklearn [improves CV score]
# 
# ## Update:
# 1. Model updated
# 2. Increased Seeds
# 

# # If you like it, Do Upvote :)

# In[1]:


import sys
sys.path.append('../iterative stratification/iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold #分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。


# In[2]:


import numpy as np
import random
import pandas as pd   #csv file processing
import matplotlib.pyplot as plt
import os   #operating system
import copy
import seaborn as sns   #statistical data visualization

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')


# In[3]:


os.listdir('../lish-moa')


# In[4]:


train_features = pd.read_csv('../lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('../lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('../lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('../lish-moa/test_features.csv')
sample_submission = pd.read_csv('../lish-moa/sample_submission.csv')

train_features_original = train_features.copy()


# In[5]:


GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
print('train_features before: ',train_features.shape)


# In[6]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


# In[7]:


# train_targets_scored.sum()[1:].sort_values()
train_features_original


# In[8]:


train_features['cp_type'].unique()
print('train_features: ',train_features.shape)


# # PCA features + Existing features

# In[9]:


# GENES
n_comp = 50

data_gene = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
data_pca_gene = (PCA(n_components=n_comp, random_state=42).fit_transform(data_gene[GENES]))
train_pca_gene = data_pca_gene[:train_features.shape[0]]  #without column name
test_pca_gene = data_pca_gene[-test_features.shape[0]:]

# print('data: ',data.shape)
# print('data2: ',data2.shape)
# print('train_pca_gene: ',train2.shape)
# print('test_pca_gene: ',test2.shape)
# print('train_features: ',train_features.shape)
# # print(train2)    


# In[10]:


train_pca_gene = pd.DataFrame(train_pca_gene, columns=[f'pca_G-{i}' for i in range(n_comp)])  # add column name
test_pca_gene = pd.DataFrame(test_pca_gene, columns=[f'pca_G-{i}' for i in range(n_comp)])

# print('train_features before: ',train_features.shape)
# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
train_features = pd.concat((train_features, train_pca_gene), axis=1)
test_features = pd.concat((test_features, test_pca_gene), axis=1)

# print('train2: ',train2.shape)
# print('test2: ',test2.shape)
# print('train_features after: ',train_features.shape)
# print('test_features',test_features.shape)
# # print(train2)


# In[11]:


# print('data_concat: ',data.shape)
# print('data_pca: ',data2.shape)
# print('train2: ',train2.shape)
# print('test2',test2.shape)

# #print(train_features)
# print('train_features: ',train_features.shape)
# print('test_features: ',test_features.shape)


# In[12]:


#CELLS
n_comp = 15

data_cell = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
data_pca_cell = (PCA(n_components=n_comp, random_state=42).fit_transform(data_cell[CELLS]))
train_pca_cell = data_pca_cell[:train_features.shape[0]]
test_pca_cell = data_pca_cell[-test_features.shape[0]:]

train_pca_cell = pd.DataFrame(train_pca_cell, columns=[f'pca_C-{i}' for i in range(n_comp)])
test_pca_cell = pd.DataFrame(test_pca_cell, columns=[f'pca_C-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
train_features = pd.concat((train_features, train_pca_cell), axis=1)
test_features = pd.concat((test_features, test_pca_cell), axis=1)
# print('train_features: ',train_features.shape)
# print('test_features: ',test_features.shape)


# # feature Selection using Variance Encoding

# In[13]:



# # data.iloc?
# mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
#           {'a': 100, 'b': 200, 'c': 300, 'd': 400},
#           {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
# df = pd.DataFrame(mydict)
# df1=df.iloc[:,2:]
# df2=df1.reset_index(drop=True)
# df3_=df[df['a']==100]
# df3_ordered=df[df['a']==100].reset_index(drop=True)
# print(df1)
# print(df2)
# print(df3_)
# print(df3_ordered)
# print(df1.shape)


# In[14]:


from sklearn.feature_selection import VarianceThreshold
#VarianceThreshold：移除方差小于threshold的特征向量

var_thresh = VarianceThreshold(threshold=0.5)  #方差阈值0.5
data_all = train_features.append(test_features)  #将train和test按行拼接

# print('train_features before threshold: ',train_features.shape)
data_all_transformed = var_thresh.fit_transform(data_all.iloc[:, 4:])  # iloc for integer location, namely 'index' for pandas

train_features_transformed = data_all_transformed[ : train_features.shape[0]]#按行截取出train
test_features_transformed = data_all_transformed[-test_features.shape[0] : ]#按行截取出test


#加入sig_id,cp_type,cp_time,cp_dose这4列内容
train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),
                              columns=['sig_id','cp_type','cp_time','cp_dose'])

train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)
# print('train_features_transformed: ',train_features_transformed.shape)

test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),
                             columns=['sig_id','cp_type','cp_time','cp_dose'])

test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

train_features


# In[15]:


#将标签数据train_targets_scored拼接在train_features右侧
train = train_features.merge(train_targets_scored, on='sig_id')

# 索引后的行的index仍然是原始的index，需要用reset_index重置index，并且将原始index那一列删除（drop）
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True) 
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

target = train[train_targets_scored.columns]


# In[16]:


print(train_targets_scored.columns)


# In[17]:


train = train.drop('cp_type', axis=1)  #axis=0，丢弃某一行；axis=1，丢弃某一列
test = test.drop('cp_type', axis=1)


# In[18]:


test


# # Binning

# In[19]:


# for col in GENES:
#     train.loc[:, f'{col}_bin'] = pd.cut(train[col], bins=3, labels=False)
#     test.loc[:, f'{col}_bin'] = pd.cut(test[col], bins=3, labels=False)

    
    


# # Distribution plots

# In[20]:


# gene 
plt.figure(figsize=(10,4))
sns.set_style("whitegrid")

gene_choice = np.random.choice(len(GENES), 6)   # 生成9个小于len(GENES)的数
for i, col in enumerate(gene_choice):
    plt.subplot(1, 6, i+1)
    if i == 0:
        plt.ylabel('frequency')
    plt.hist(train_features_original.loc[:, GENES[col]],bins=100, color='orange')
    plt.title(GENES[col])
    


# In[21]:


# pca gene
train_pca_gene_cols = train_pca_gene.columns

plt.figure(figsize=(10,4))
sns.set_style("whitegrid")

pca_gene_choice = np.random.choice(len(train_pca_gene_cols), 6)   # 生成9个小于len(GENES)的数
for i, col in enumerate(pca_gene_choice):
    plt.subplot(1, 6, i+1)
    if i == 0:
        plt.ylabel('frequency')
    plt.hist(train_pca_gene.loc[:, train_pca_gene_cols[col]],bins=100, color='orange')
    plt.title(train_pca_gene_cols[col])


# In[22]:


# cell 
plt.figure(figsize=(10,4))
sns.set_style("whitegrid")

cell_choice = np.random.choice(len(CELLS), 6)   # 生成9个小于len(GENES)的数
for i, col in enumerate(cell_choice):
    plt.subplot(1,6, i+1)
    if i == 0:
        plt.ylabel('frequency')
    plt.hist(train_features_original.loc[:, CELLS[col]],bins=100, color='orange')
    plt.title(CELLS[col])


# In[23]:


# pca cell
train_pca_cell_cols = train_pca_cell.columns

plt.figure(figsize=(10,4))
sns.set_style("whitegrid")

pca_cell_choice = np.random.choice(len(train_pca_cell_cols), 6)   # 生成9个小于len(GENES)的数
for i, col in enumerate(pca_cell_choice):
    plt.subplot(1,6, i+1)
    if i == 0:
        plt.ylabel('frequency')
    plt.hist(train_pca_cell.loc[:, train_pca_cell_cols[col]],bins=100, color='orange')
    plt.title(train_pca_cell_cols[col])


# # [Naive] Outlier Removal

# In[24]:



# train_ = train.copy() [Didn't wanted to actually normalize, so created a copy and normalized that for further calculation]
# for col in GENES:
    
# #     train_[col] = (train[col]-np.mean(train[col])) / (np.std(train[col]))
    
#     mean = train_[col].mean()
#     std = train_[col].std()

#     std_r = mean + 4*std
#     std_l = mean - 4*std

#     drop = train_[col][(train_[col]>std_r) | (train_[col]<std_l)].index.values

# train = train.drop(drop).reset_index(drop=True)
# # folds = folds.drop(drop).reset_index(drop=True)
# target = target.drop(drop).reset_index(drop=True)


# # PCA

# In[25]:


# n_comp = 50

# data = pd.concat([pd.DataFrame(train[CELLS]), pd.DataFrame(test[CELLS])])
# data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[CELLS]))
# train2 = data2[:train.shape[0]]; test2 = data2[train.shape[0]:]

# train2 = pd.DataFrame(train2, columns=[f'c-{i}' for i in range(n_comp)])
# test2 = pd.DataFrame(test2, columns=[f'c-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
# train = train.drop(columns=drop_cols)
# test = test.drop(columns=drop_cols)


# In[26]:


target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

print(len(target_cols))


# # CV folds

# In[27]:


folds = train.copy()

mskf = MultilabelStratifiedKFold(n_splits=5)

for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
    folds.loc[v_idx, 'kfold'] = int(f)#添加一列'kfold'，作为fold的标记

folds['kfold'] = folds['kfold'].astype(int)
folds


# In[28]:


# print(train.shape)
# print(folds.shape)
# print(test.shape)
# print(target.shape)
# print(sample_submission.shape)


# # Dataset Classes

# In[29]:


class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
        }
        return dct
    
class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct
    


# In[30]:


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
#         print(inputs.shape)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    
    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []
    
    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
        
        preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    preds = np.concatenate(preds)
    
    return preds
   
    


# # Model

# In[31]:


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.5)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x


# # Preprocessing steps

# In[32]:


def process_data(data):
    
    data = pd.get_dummies(data, columns=['cp_time','cp_dose'])#对cp_time,cp_dose两列进行one-hot 编码，dummy 表示除去任意一个状态
#     data.loc[:, 'cp_time'] = data.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})
#     data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

# --------------------- Normalize ---------------------
#     for col in GENES:
#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))
    
#     for col in CELLS:
#         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))
    
#--------------------- Removing Skewness ---------------------
#     for col in GENES + CELLS:
#         if(abs(data[col].skew()) > 0.75):
            
#             if(data[col].skew() < 0): # neg-skewness
#                 data[col] = data[col].max() - data[col] + 1
#                 data[col] = np.sqrt(data[col])
            
#             else:
#                 data[col] = np.sqrt(data[col])
    
    return data


# In[33]:


feature_cols = [c for c in process_data(folds).columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]
len(feature_cols)


# In[34]:


# HyperParameters

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False

num_features=len(feature_cols)
num_targets=len(target_cols)
hidden_size=1024


# # Single fold training

# In[35]:


def run_training(fold, seed):
    
    seed_everything(seed)
    
    train = process_data(folds)#one-hot encoding
    test_ = process_data(test)
    
    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index
    
    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)
    
    x_train, y_train  = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid =  valid_df[feature_cols].values, valid_df[target_cols].values
    
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0
    
    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    best_loss = np.inf
    
    for epoch in range(EPOCHS):
        
        train_loss = train_fn(model, optimizer,scheduler, loss_fn, trainloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), f"FOLD{fold}_.pth")
        
        elif(EARLY_STOP == True):
            
            early_step += 1
            if (early_step >= early_stopping_steps):
                break
            
    
    #--------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))
    model.to(DEVICE)
    
    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, DEVICE)
    
    return oof, predictions


# In[36]:


def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))
    
    for fold in range(NFOLDS):
        oof_, pred_ = run_training(fold, seed)
        
        predictions += pred_ / NFOLDS
        oof += oof_
        
    return oof, predictions


# In[37]:


# Averaging on multiple SEEDS

#SEED=[0]
SEED = [0, 1, 2, 3 ,4, 5]
oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))

for seed in SEED:
    
    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)



# In[38]:


print(train.shape)
print(test.shape)


# In[39]:


train[target_cols] = oof
test[target_cols] = predictions


# In[40]:


# test['atp-sensitive_potassium_channel_antagonist'] = 0.0
# test['erbb2_inhibitor'] = 0.0

# train['atp-sensitive_potassium_channel_antagonist'] = 0.0
# train['erbb2_inhibitor'] = 0.0
test


# In[41]:


train_targets_scored


# In[42]:


len(target_cols)


# In[43]:


valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)


y_true = train_targets_scored[target_cols].values
y_pred = valid_results[target_cols].values

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]
    
print("CV log_loss: ", score)
    


# In[44]:


sub = sample_submission.drop(columns=target_cols).merge(test[['sig_id']+target_cols], on='sig_id', how='left').fillna(0)
sub.to_csv('submission.csv', index=False)


# In[45]:


sub.shape


# ## Things that can improve your CV even further:
# 1. Increasing SEEDS
# 2. Feature Selection over GENES/CELLS columns
# 3. Model Hyperparameter Tuning
# 4. Removing Skewness from GENES/CELLS columns [Comment below if it helps]
# 5. PCA........................................[Comment below if it helps]
# 
