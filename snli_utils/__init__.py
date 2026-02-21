import io
import time

import dill
import urllib.request
import zipfile

import torchvision
from torch.utils.data import Dataset


def write_to_file(name,text,mode='a'):
    with open(name, mode) as myfile:
        myfile.write(text+'\n')

import torch
import os
import sys
import numpy as np
import pandas as pd
import torchtext

import os
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import regex as re
from string import punctuation
import math
from io import StringIO

import nltk
# # nltk.download("omw-1.4")
#nltk.download('punkt')
#nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

import swifter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

verbose=True
cache_preprocessing=True
additional_prefix=''
data_file_name='../data/Preprocessed_SNLI_dataloaders'
data_file=''

# df_train=pd.DataFrame()
# df_test=pd.DataFrame()
train_loader :torch.utils.data.DataLoader= None#torch.utils.data.DataLoader({},shuffle=True)
test_loader:torch.utils.data.DataLoader=None


def get_SNLI_data():
    train_file_name = additional_prefix+'../data/snli_1.0_train.csv'
    test_file_name = additional_prefix+'../data/snli_1.0_test.csv'

    if not os.path.isfile(train_file_name) or not os.path.isfile(test_file_name):
        snli_url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
        # dload.save_unzip(snli_url, "../data/snli_1.0")
        # url = 'http://www.gutenberg.lib.md.us/4/8/8/2/48824/48824-8.zip'
        filehandle, _ = urllib.request.urlretrieve(snli_url)
        zip_file_object = zipfile.ZipFile(filehandle, 'r')
        #first_file = zip_file_object.namelist()[0]
        file_train = zip_file_object.open('snli_1.0/snli_1.0_train.txt')
        file_test=zip_file_object.open('snli_1.0/snli_1.0_test.txt')
        #content = file.read()
        # train=file_train.read()
        # test=file_test.read()
        with open(train_file_name,'wb') as f:
            f.write(file_train.read())
        with open(test_file_name,'wb') as f:
            f.write(file_test.read())
        file_train.seek(0)
        file_test.seek(0)
    #     return file_train.read(),file_test.read()
    # with open(train_file_name, 'rb') as f:
    #     with open(test_file_name, 'rb') as f2:
    #         return f.read(),f2.read()

#train_data,test_data=get_SNLI_data()

#df=pd.read_csv(StringIO(str(train_data)),sep='\t')
# df = pd.read_csv(io.StringIO("csv string"))

#df=pd.read_csv('../data/snli_1.0_train.csv', on_bad_lines='skip')
# if __name__ == '__main__':
#     get_SNLI_data()
#     df_train=pd.read_csv(train_file_name,sep='\t')
#     df_test=pd.read_csv(test_file_name,sep='\t')
#     # df_train=df_train.drop([],axis=1)
#     # df_test=df_test.drop([],axis=1)
#     # df_train=df_train.drop_duplicates()
#     # df_test=df_test.drop_duplicates()
#     #concatenated=[(s1 if s1[-1]=='.' else s1+'.')+' '+s2 for s1,s2 in zip(df_train['sentence1'],df_train['sentence2'])]
#
#     #concatenated=[(str(s1) if str(s1)[-1]=='.' else str(s1)+'.')+' '+str(s2) for s1,s2 in zip(df_train['sentence1'],df_train['sentence2'])]
#     #df_train['concatenated']=(df_train['sentence1'] if df_train['sentence1'][-1]=='.' else df_train['sentence1']+'.')+' '+df_train['sentence2']
#     #concatenate sentences
#     df_train['concatenated']=[(str(s1) if str(s1)[-1]=='.' else str(s1)+'.')+' '+str(s2) for s1,s2 in zip(df_train['sentence1'],df_train['sentence2'])]
#     df_test['concatenated']=[(str(s1) if str(s1)[-1]=='.' else str(s1)+'.')+' '+str(s2) for s1,s2 in zip(df_test['sentence1'],df_test['sentence2'])]
#
#     #to lowercase
#     print(df_train.head())
#     df_train['concatenated']=df_train['concatenated'].swifter.apply(lambda x:x.lower())
#     df_test['concatenated']=df_test['concatenated'].swifter.apply(lambda x:x.lower())
#     print(df_train.head())
#     print(df_test.head())
#     print(df_train['gold_label'][0]+': '+df_train['concatenated'][0])
#     print(df_train['gold_label'][1]+': '+df_train['concatenated'][1])
#     print(df_train['gold_label'][2]+': '+df_train['concatenated'][2])
#
#     #delete rows without gold label
#     print(df_train.describe())
#     df_train.drop(df_train[df_train['gold_label']=='-'].index,inplace=True)
#     print(df_train.describe())
#
#     print(df_test.describe())
#     df_test.drop(df_test[df_test['gold_label']=='-'].index,inplace=True)
#     print(df_test.describe())
#
#     #string labels to numbers
#     label_dict={'contradiction':0,'neutral':1,'entailment':2}
#     df_train['gold_label']=df_train['gold_label'].swifter.apply(lambda x:label_dict[x])
#     print(df_train.describe())
#     print(df_train.gold_label.dtype)
#
#     df_test['gold_label']=df_test['gold_label'].swifter.apply(lambda x:label_dict[x])
#
#     #drop unused columns
#     columns=list(df_train.columns.values)
#     columns.remove('gold_label')
#     columns.remove('concatenated')
#     df_train=df_train.drop(columns,axis=1)
#     # columns=list(df_test.columns.values)
#     # columns.remove('gold_label')
#     # columns.remove('concatenated')
#     df_test=df_test.drop(columns,axis=1)
#     print(df_train.columns.values)
#     print(df_test.columns.values)
#
#     d_type = float
#     #labels to logic array
#     # def int_to_logic_array(num:int):
#     #     return np.asarray(np.arange(3) == num).astype(type)
#     def int_to_logic_array(num: int):
#         z = np.zeros(3, dtype=d_type)
#         z[num] = 1
#         return z
#     print(int_to_logic_array(1))
#
#     #labels=df_train['gold_label'].apply(lambda x:int_to_logic_array(x,d_type)).to_numpy()
#     labels=df_train['gold_label'].swifter.apply(int_to_logic_array).to_numpy()
#     #print(np.vstack(labels).astype(d_type).shape)
#     print(np.vstack(labels).shape)
#
#     test_labels = df_test['gold_label'].swifter.apply(int_to_logic_array).to_numpy()
#     # print(np.vstack(labels).astype(d_type).shape)
#     print(np.vstack(test_labels).shape)
#
#
#     #tokenization
#     def tokenize(series):
#         return word_tokenize(series)
#
#     df_train['tokens'] = df_train['concatenated'].swifter.apply(tokenize)
#     df_test['tokens'] = df_test['concatenated'].swifter.apply(tokenize)
#
#     def get_len(series):
#         return len(series)
#
#     df_train['token_len'] = df_train['tokens'].swifter.apply(get_len)
#     df_test['token_len'] = df_test['tokens'].swifter.apply(get_len)
#
#     print(df_train.head())
#     print(df_train['tokens'].head())
#     print(df_train['tokens'])
#
#     #padding
#     MAX_LEN=df_test['token_len'].max()#=73
#     def pad_token(series):
#         if len(series) < MAX_LEN:
#             series.extend(['<END>'] * (MAX_LEN - len(series)))
#             return series
#         else:
#             return series[:MAX_LEN]  # series[:MAX_LEN]#series[len(series)-MAX_LEN:len(series)]
#     df_train['padded_tokens']=df_train['tokens'].swifter.apply(pad_token)
#     df_test['padded_tokens']=df_test['tokens'].swifter.apply(pad_token)
#
#
#
#     unique_words = set()
#     for tokens in list(df_train['padded_tokens'].values)+list(df_test['padded_tokens'].values):
#         unique_words.update(tokens)
#
#     print('Count of Unique words:', len(unique_words))
#
#
#     word2idx = {}
#     for word in unique_words:
#         word2idx[word] = len(word2idx)
#     if not '<END>' in word2idx:
#         word2idx['<END>'] = len(word2idx)
#
#     word_embeddings = np.random.rand(len(word2idx), 50)
#     with open('./glove.6B/glove.6B.50d.txt', 'r',encoding='utf8') as embeds:
#         embeddings = embeds.read()
#         embeddings = embeddings.split('\n')[:-2]
#
#     for token_idx, token_embed in enumerate(embeddings):
#         token = token_embed.split()[0]
#         if token in word2idx:
#             word_embeddings[word2idx[token]] = [float(val) for val in token_embed.split()[1:]]
#
#     print(f'Word embeddings for word {list(word2idx.keys())[300]}:', word_embeddings[300])
#
#     ###############
#     #print(df_train[['lemma_tokens', 'sentiment']].head())
#     df_train = df_train.reset_index()
#     #print(df_test[['lemma_tokens', 'sentiment']].head())
#     df_test = df_test.reset_index()
#
#     # embedding_array = np.array(
#     #             [np.array([np.array([word_embeddings[word2idx[word]]], dtype=d_type) for word in review]).squeeze() for
#     #              review in
#     #              df_train['padded_tokens']])  # .squeeze()
#     def vectorize_text(text):
#         return np.array([np.array([word_embeddings[word2idx[word]]], dtype=d_type) for word in text])
#     embedding_array=np.vstack(df_train['padded_tokens'].swifter.apply(vectorize_text).to_numpy()).reshape((-1,1,MAX_LEN,50))
#     # shape=list(embedding_array.shape)
#     # shape.insert(1,1)
#     # embedding_array=embedding_array.reshape(shape)#[1]+list(embedding_array.shape))
#     # #df_train['embeddings'] = embedding_array
#     train_loader = torch.utils.data.DataLoader(list(zip(embedding_array,labels)), shuffle=True)
#
#     # test_embedding_array = np.array(
#     #     [np.array([np.array([word_embeddings[word2idx[word]]], dtype=d_type) for word in review]).squeeze() for
#     #      review in
#     #      df_test['padded_tokens']])  # .squeeze()
#     test_embedding_array = np.vstack(df_test['padded_tokens'].swifter.apply(vectorize_text).to_numpy()).reshape((-1, 1,MAX_LEN, 50))
#     # test_shape = list(test_embedding_array.shape)
#     # test_shape.insert(1, 1)
#     # test_embedding_array = test_embedding_array.reshape(test_shape)  # [1]+list(embedding_array.shape))
#     # #df_test['embeddings'] = test_embedding_array
#     test_loader = torch.utils.data.DataLoader(list(zip(test_embedding_array,test_labels)), shuffle=True)
#
#     data_file=data_file_name+'_'+'float'+'.pkl'
#     if cache_preprocessing:
#         with open(data_file, "wb") as dill_file:
#             #dill.dump(df_train, dill_file)
#             dill.dump((train_loader,test_loader), dill_file)


def get_preprocessed_SNLI(d_type=torch.float)->(torch.utils.data.DataLoader,torch.utils.data.DataLoader):
    train_file_name = additional_prefix + '../data/snli_1.0_train.csv'
    test_file_name = additional_prefix + '../data/snli_1.0_test.csv'

    d_type=str(d_type).split(".",1)[1]
    global data_file
    data_file=additional_prefix+data_file_name+'_'+d_type+'.pkl'
    global train_loader
    global test_loader
    if os.path.isfile(data_file):
        with open(data_file, "rb") as dill_file:
            #df_train=dill.load(dill_file)
            train_loader,test_loader=dill.load(dill_file)

    if train_loader is None or len(train_loader)==0:# or test_loader is None or len(test_loader)==0:#df_train.empty:
        get_SNLI_data()
        df_train = pd.read_csv(train_file_name, sep='\t')
        df_test = pd.read_csv(test_file_name, sep='\t')
        # df_train=df_train.drop([],axis=1)
        # df_test=df_test.drop([],axis=1)
        # df_train=df_train.drop_duplicates()
        # df_test=df_test.drop_duplicates()
        # concatenated=[(s1 if s1[-1]=='.' else s1+'.')+' '+s2 for s1,s2 in zip(df_train['sentence1'],df_train['sentence2'])]

        # concatenated=[(str(s1) if str(s1)[-1]=='.' else str(s1)+'.')+' '+str(s2) for s1,s2 in zip(df_train['sentence1'],df_train['sentence2'])]
        # df_train['concatenated']=(df_train['sentence1'] if df_train['sentence1'][-1]=='.' else df_train['sentence1']+'.')+' '+df_train['sentence2']
        # concatenate sentences
        df_train['concatenated'] = [(str(s1) if str(s1)[-1] == '.' else str(s1) + '.') + ' ' + str(s2) for s1, s2 in
                                    zip(df_train['sentence1'], df_train['sentence2'])]
        df_test['concatenated'] = [(str(s1) if str(s1)[-1] == '.' else str(s1) + '.') + ' ' + str(s2) for s1, s2 in
                                   zip(df_test['sentence1'], df_test['sentence2'])]

        # to lowercase
        print(df_train.head())
        df_train['concatenated'] = df_train['concatenated'].swifter.apply(lambda x: x.lower())
        df_test['concatenated'] = df_test['concatenated'].swifter.apply(lambda x: x.lower())
        print(df_train.head())
        print(df_test.head())
        print(df_train['gold_label'][0] + ': ' + df_train['concatenated'][0])
        print(df_train['gold_label'][1] + ': ' + df_train['concatenated'][1])
        print(df_train['gold_label'][2] + ': ' + df_train['concatenated'][2])

        # delete rows without gold label
        print(df_train.describe())
        df_train.drop(df_train[df_train['gold_label'] == '-'].index, inplace=True)
        print(df_train.describe())

        print(df_test.describe())
        df_test.drop(df_test[df_test['gold_label'] == '-'].index, inplace=True)
        print(df_test.describe())

        # string labels to numbers
        label_dict = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        df_train['gold_label'] = df_train['gold_label'].swifter.apply(lambda x: label_dict[x])
        print(df_train.describe())
        print(df_train.gold_label.dtype)

        df_test['gold_label'] = df_test['gold_label'].swifter.apply(lambda x: label_dict[x])

        # drop unused columns
        columns = list(df_train.columns.values)
        columns.remove('gold_label')
        columns.remove('concatenated')
        df_train = df_train.drop(columns, axis=1)
        # columns=list(df_test.columns.values)
        # columns.remove('gold_label')
        # columns.remove('concatenated')
        df_test = df_test.drop(columns, axis=1)
        print(df_train.columns.values)
        print(df_test.columns.values)

        #d_type = float

        # labels to logic array
        # def int_to_logic_array(num:int):
        #     return np.asarray(np.arange(3) == num).astype(type)
        def int_to_logic_array(num: int):
            z = np.zeros(3, dtype=d_type)
            z[num] = 1
            return z

        print(int_to_logic_array(1))

        # labels=df_train['gold_label'].apply(lambda x:int_to_logic_array(x,d_type)).to_numpy()
        labels = df_train['gold_label'].swifter.apply(int_to_logic_array).to_numpy()
        # print(np.vstack(labels).astype(d_type).shape)
        print(np.vstack(labels).shape)

        test_labels = df_test['gold_label'].swifter.apply(int_to_logic_array).to_numpy()
        # print(np.vstack(labels).astype(d_type).shape)
        print(np.vstack(test_labels).shape)

        # tokenization
        def tokenize(series):
            return word_tokenize(series)

        df_train['tokens'] = df_train['concatenated'].swifter.apply(tokenize)
        df_test['tokens'] = df_test['concatenated'].swifter.apply(tokenize)

        def get_len(series):
            return len(series)

        df_train['token_len'] = df_train['tokens'].swifter.apply(get_len)
        df_test['token_len'] = df_test['tokens'].swifter.apply(get_len)

        print(df_train.head())
        print(df_train['tokens'].head())
        print(df_train['tokens'])

        # padding
        MAX_LEN = df_test['token_len'].max()  # =73

        def pad_token(series):
            if len(series) < MAX_LEN:
                series.extend(['<END>'] * (MAX_LEN - len(series)))
                return series
            else:
                return series[:MAX_LEN]  # series[:MAX_LEN]#series[len(series)-MAX_LEN:len(series)]

        df_train['padded_tokens'] = df_train['tokens'].swifter.apply(pad_token)
        df_test['padded_tokens'] = df_test['tokens'].swifter.apply(pad_token)

        unique_words = set()
        for tokens in list(df_train['padded_tokens'].values) + list(df_test['padded_tokens'].values):
            unique_words.update(tokens)

        print('Count of Unique words:', len(unique_words))

        word2idx = {}
        for word in unique_words:
            word2idx[word] = len(word2idx)
        if not '<END>' in word2idx:
            word2idx['<END>'] = len(word2idx)

        word_embeddings = np.random.rand(len(word2idx), 50)
        with open(additional_prefix+'../glove.6B/glove.6B.50d.txt', 'r', encoding='utf8') as embeds:
            embeddings = embeds.read()
            embeddings = embeddings.split('\n')[:-2]

        for token_idx, token_embed in enumerate(embeddings):
            token = token_embed.split()[0]
            if token in word2idx:
                word_embeddings[word2idx[token]] = [float(val) for val in token_embed.split()[1:]]

        print(f'Word embeddings for word {list(word2idx.keys())[300]}:', word_embeddings[300])

        ###############
        # print(df_train[['lemma_tokens', 'sentiment']].head())
        df_train = df_train.reset_index()
        # print(df_test[['lemma_tokens', 'sentiment']].head())
        df_test = df_test.reset_index()

        # embedding_array = np.array(
        #             [np.array([np.array([word_embeddings[word2idx[word]]], dtype=d_type) for word in review]).squeeze() for
        #              review in
        #              df_train['padded_tokens']])  # .squeeze()
        def vectorize_text(text):
            return np.array([np.array([word_embeddings[word2idx[word]]], dtype=d_type) for word in text])

        embedding_array = np.vstack(df_train['padded_tokens'].swifter.apply(vectorize_text).to_numpy()).reshape(
            (-1, 1, MAX_LEN, 50))
        # shape=list(embedding_array.shape)
        # shape.insert(1,1)
        # embedding_array=embedding_array.reshape(shape)#[1]+list(embedding_array.shape))
        # #df_train['embeddings'] = embedding_array
        train_loader = torch.utils.data.DataLoader(list(zip(embedding_array, labels)), shuffle=True)

        # test_embedding_array = np.array(
        #     [np.array([np.array([word_embeddings[word2idx[word]]], dtype=d_type) for word in review]).squeeze() for
        #      review in
        #      df_test['padded_tokens']])  # .squeeze()
        test_embedding_array = np.vstack(df_test['padded_tokens'].swifter.apply(vectorize_text).to_numpy()).reshape(
            (-1, 1, MAX_LEN, 50))
        # test_shape = list(test_embedding_array.shape)
        # test_shape.insert(1, 1)
        # test_embedding_array = test_embedding_array.reshape(test_shape)  # [1]+list(embedding_array.shape))
        # #df_test['embeddings'] = test_embedding_array
        test_loader = torch.utils.data.DataLoader(list(zip(test_embedding_array, test_labels)), shuffle=True)

        #data_file = data_file_name + '_' + 'float' + '.pkl'
        if cache_preprocessing:
            with open(data_file, "wb") as dill_file:
                # dill.dump(df_train, dill_file)
                dill.dump((train_loader, test_loader), dill_file)
    return (train_loader,test_loader)

def get_model_layers(d_type=torch.float):
    # return [torch.nn.Conv2d(1,16,(1,50),stride=1,padding='valid',dtype=d_type),
    #         torch.nn.ELU(),
    #         #torch.permute(),
    #         #torch.nn.Conv1d(16,16,3,stride=1,padding='valid'),
    #         torch.nn.Conv2d(16,16,(3,1),stride=2,padding='valid',dtype=d_type),
    #         # torch.nn.Tanh(),
    #         # torch.nn.Conv2d(16, 16, (3, 1), stride=2, padding='valid'),
    #         # torch.nn.Tanh(),
    #         # torch.nn.Conv3d(16, 16, (3,3,1), stride=(1,2,1),padding='valid'),
    #         # torch.nn.Tanh(),
    #         # torch.nn.Conv3d(16, 16, (3, 3, 1), stride=(1, 2, 1), padding='valid'),
    #         # torch.nn.Tanh(),
    #         # torch.nn.Conv3d(16, 16, (3, 3, 1), stride=(1, 2, 1), padding='valid'),
    #         torch.nn.Tanh(),]+\
    #        [
    #         torch.nn.Conv2d(16, 16, (3, 1), stride=1, padding='valid',dtype=d_type),torch.nn.Tanh(),]*20+\
    #        [torch.nn.Flatten(),
    #         torch.nn.Linear(320, 32,dtype=d_type),
    #         # torch.nn.Tanh(),
    #         # torch.nn.Linear(32, 32),
    #         # torch.nn.Tanh(),
    #         # torch.nn.Linear(32, 32),
    #         torch.nn.Tanh(),
    #         torch.nn.Linear(32, 24,dtype=d_type),
    #         torch.nn.Tanh(),
    #         torch.nn.Linear(24, 16,dtype=d_type),
    #         torch.nn.Tanh(),
    #         torch.nn.Linear(16, 12,dtype=d_type),
    #         torch.nn.Tanh(),
    #         torch.nn.Linear(12, 8,dtype=d_type),
    #         torch.nn.Tanh(),
    #         torch.nn.Linear(8, 6,dtype=d_type),
    #         torch.nn.Tanh(),
    #         torch.nn.Linear(6, 4,dtype=d_type),
    #         torch.nn.Tanh(),
    #         torch.nn.Linear(4,2,dtype=d_type)
    #         ]
    return [torch.nn.Flatten(),torch.nn.Linear(73*50,1,dtype=d_type)]
    layers=[torch.nn.Conv2d(1, 16, (1, 50), stride=1, padding='valid', dtype=d_type),
     torch.nn.ELU(),
     # torch.permute(),
     # torch.nn.Conv1d(16,16,3,stride=1,padding='valid'),
     torch.nn.Conv2d(16, 16, (3, 1), stride=2, padding='valid', dtype=d_type),
     # torch.nn.Tanh(),
     # torch.nn.Conv2d(16, 16, (3, 1), stride=2, padding='valid'),
     # torch.nn.Tanh(),
     # torch.nn.Conv3d(16, 16, (3,3,1), stride=(1,2,1),padding='valid'),
     # torch.nn.Tanh(),
     # torch.nn.Conv3d(16, 16, (3, 3, 1), stride=(1, 2, 1), padding='valid'),
     # torch.nn.Tanh(),
     # torch.nn.Conv3d(16, 16, (3, 3, 1), stride=(1, 2, 1), padding='valid'),
     torch.nn.Tanh(), ]
    for _ in range(20):
        layers+=(lambda:[torch.nn.Conv2d(16, 16, (3, 1), stride=1, padding='valid', dtype=d_type), torch.nn.Tanh(), ])()
    layers+=[torch.nn.Flatten(),
     torch.nn.Linear(320, 32, dtype=d_type),
     # torch.nn.Tanh(),
     # torch.nn.Linear(32, 32),
     # torch.nn.Tanh(),
     # torch.nn.Linear(32, 32),
     torch.nn.Tanh(),
     torch.nn.Linear(32, 24, dtype=d_type),
     torch.nn.Tanh(),
     torch.nn.Linear(24, 16, dtype=d_type),
     torch.nn.Tanh(),
     torch.nn.Linear(16, 12, dtype=d_type),
     torch.nn.Tanh(),
     torch.nn.Linear(12, 8, dtype=d_type),
     torch.nn.Tanh(),
     torch.nn.Linear(8, 6, dtype=d_type),
     torch.nn.Tanh(),
     torch.nn.Linear(6, 4, dtype=d_type),
     torch.nn.Tanh(),
     torch.nn.Linear(4, 2, dtype=d_type)
     ]
    return layers


class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    # def __getitem__(self, index):
    #     x, y = self.subset[index]
    #     if self.transform:
    #         x = self.transform(x)
    #         y=self.transform(y)
    #     return x, y
    def __getitem__(self, index):
        x, y = self.subset[index]
        return self.transform(x),self.transform(y)

    def __len__(self):
        return len(self.subset)
if __name__=='__main__':
    additional_prefix='../'

    d_type=torch.float32#torch.float64
    train_loader, test_loader=get_preprocessed_SNLI(d_type=d_type)
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, shuffle=True, batch_size=128)
    test_loader = torch.utils.data.DataLoader(test_loader.dataset, shuffle=True, batch_size=128)

    transform_dtype = True
    if transform_dtype and d_type!=torch.float64:
        d_type=torch.float64
        # convert_to_datatype = lambda x: x.to(d_type)
        _d_type=float if d_type==torch.float64 else np.float32
        convert_to_datatype = lambda x: x.astype(_d_type)
        transform = torchvision.transforms.Compose([
                #torchvision.transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,))
                convert_to_datatype
            ])
        train_loader = torch.utils.data.DataLoader(MyDataset(train_loader.dataset,transform), shuffle=True, batch_size=128)
        test_loader = torch.utils.data.DataLoader(MyDataset(test_loader.dataset,transform), shuffle=True, batch_size=128)

    #x=list(enumerate(test_loader))
    #print(np.array(x[0][1][0]).shape)
    import main_linear_avggrad_implementation as main
    import torchsummary

    model = main.NN(get_model_layers(d_type=d_type))
    # #print(model(torch.tensor([list(enumerate(test_loader))[0][1][1]])))
    # x=list(enumerate(test_loader))[0][1][0]
    # print([1]+list(x.shape))
    # x=x.reshape([1]+list(x.shape))

    #print(model(x))
    if d_type==torch.float:
        torchsummary.summary(model,(1,73,50))
    #print(list(enumerate(train_loader)))
    #print(list(enumerate(test_loader)))
    #device='cpu'
    a=set()
    #train_loader.batch_size=128
    for batch_idx, sample in enumerate(train_loader):
        #data, target = data.to(device), target.to(device)
        a.add(len(sample[0][0][0]))
        model(sample[0])
        #print(sample)
    print(a)

    a=set()
    for batch_idx, sample in enumerate(test_loader):
        #data, target = data.to(device), target.to(device)
        a.add(len(sample[0][0][0]))
        model(sample[0])
        #print(sample)
    print(a)


