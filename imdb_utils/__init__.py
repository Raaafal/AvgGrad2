import dill
import torchvision


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

import nltk
# # nltk.download("omw-1.4")
#nltk.download('punkt')
#nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

verbose=True
cache_preprocessing=True
additional_prefix=''
data_file_name='../data/Preprocessed_IMDB_dataloaders'
data_file=''

# df_train=pd.DataFrame()
# df_test=pd.DataFrame()
train_loader :torch.utils.data.DataLoader= None#torch.utils.data.DataLoader({},shuffle=True)
test_loader:torch.utils.data.DataLoader=None

def get_preprocessed_IMDB(d_type=torch.float)->(torch.utils.data.DataLoader,torch.utils.data.DataLoader):
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
        def get_IMDB_from_torchtext():
            train_iter = torchtext.datasets.IMDB('../data',split='train')
            test_iter = torchtext.datasets.IMDB('../data',split='test')

            labels, reviews = [], []
            for label, line in train_iter:
                #assert label in ('pos', 'neg')
                labels.append(label)
                reviews.append(line)
                # if label==2:
                #     print('2')
            #print(set(labels))

            df_train = pd.DataFrame({'sentiment': np.array(labels)-1, 'review': reviews})
            #df_train['sentiment'] = df_train['sentiment'].map({'pos': 1, 'neg': 0})
            if verbose:
                print('original df_train.shape: ', df_train.shape)
        #     df_train = df_train.drop_duplicates()
        #     print('after drop_duplicates, df_train.shape: ', df_train.shape)

            labels, reviews = [], []
            for label, line in test_iter:
                #assert label in ('pos', 'neg')
                labels.append(label)
                reviews.append(line)
                # if label==2:
                #     print('2')
            df_test = pd.DataFrame({'sentiment': np.array(labels)-1, 'review': reviews})
            #df_test['sentiment'] = df_test['sentiment'].map({'pos': 1, 'neg': 0})
            if verbose:
                print('original df_test.shape: ', df_test.shape)
            #df_test = df_test.drop_duplicates()
            #print('after drop_duplicates, df_test.shape: ', df_test.shape)
            #print(set(labels))
            return df_train, df_test

        df_train, df_test = get_IMDB_from_torchtext()
        # df_train.to_csv('../data/imdb_train.csv', index=False)
        # df_test.to_csv('../data/imdb_test.csv', index=False)
        if verbose:
            print(df_train.shape, df_test.shape)
        #print(df_train[df_train['review'].duplicated() == True])


        ############################################################CLEANING
        df_train.drop_duplicates(subset='review', inplace=True)
        if verbose:
            print(df_train.describe())
        df_test.drop_duplicates(subset='review', inplace=True)
        if verbose:
            print(df_test.describe())

        def remove_punc(series):
            temp = re.sub(f'[{punctuation}]', '', series)
            temp = re.sub(' br br ',' ', temp)
            temp = re.sub(' n ',' ', temp)
            return temp

        if verbose:
            print(df_train[['review']].head())
            print(df_test[['review']].head())

        df_train['review'] = df_train['review'].apply(remove_punc)
        df_test['review'] = df_test['review'].apply(remove_punc)

        if verbose:
            #print(df_train[['review']].head())

            print(df_train[['review','sentiment']])
            print(df_test[['review','sentiment']])

        def remove_stop(series):
            return ' '.join([x.lower() for x in series.split(' ') if x.lower() not in STOPWORDS])

        df_train['review'] = df_train['review'].apply(remove_stop)
        df_test['review'] = df_test['review'].apply(remove_stop)
        if verbose:
            print(df_train[['review']].head())
            print(df_test[['review']].head())

        ################################################TOKENIZATION

        def tokenize(series):
            return word_tokenize(series)

        df_train['tokens'] = df_train['review'].apply(tokenize)
        df_test['tokens'] = df_test['review'].apply(tokenize)

        def get_len(series):
            return len(series)

        df_train['token_len'] = df_train['tokens'].apply(get_len)
        df_test['token_len'] = df_test['tokens'].apply(get_len)

        print(df_train[['tokens','token_len']].head())
        print(df_train.describe())

        print(df_test[['tokens', 'token_len']].head())
        print(df_test.describe())

        ########################################PADDING

        MAX_LEN = math.ceil(df_train.describe()['token_len']['mean'])
        print()
        print(MAX_LEN)


        def pad_token(series):
            if len(series) < MAX_LEN:
                series.extend(['<END>']*(MAX_LEN-len(series)))
                return series
            else:
                return series[len(series)-MAX_LEN:len(series)]#series[:MAX_LEN]#series[len(series)-MAX_LEN:len(series)]

        df_train['paded_tokens'] = df_train['tokens'].apply(pad_token)
        df_test['paded_tokens'] = df_test['tokens'].apply(pad_token)
        print(df_train['paded_tokens'].values[10])
        print(df_test['paded_tokens'].values[10])
        #print(df_train[['tokens','paded_tokens']])

        ##############################################################LEMMATIZING
        lemmatizer = WordNetLemmatizer()
        def lemma(series):
            return [lemmatizer.lemmatize(word) for word in series]

        df_train['lemma_tokens'] = df_train['paded_tokens'].apply(lemma)
        df_test['lemma_tokens'] = df_test['paded_tokens'].apply(lemma)

        print(df_train[['paded_tokens','lemma_tokens']])
        print(df_test[['paded_tokens', 'lemma_tokens']])

        #############################################################STEMMING

        # stemmer = PorterStemmer()
        #
        # def stem(series):
        #     return [stemmer.stem(word) for word in series]
        #
        # df_train['stem_tokens'] = df_train['paded_tokens'].apply(stem)
        # df_test['stem_tokens'] = df_test['paded_tokens'].apply(stem)
        #
        # print(df_train[['paded_tokens','stem_tokens']])

        ######################################################Part of Speech Tagging

        # def pos_t(series):
        #     return nltk.pos_tag(series, tagset='universal')
        #
        # df_train['pos_tag_tokens'] = df_train['paded_tokens'].apply(pos_t)
        # df_test['pos_tag_tokens'] = df_test['paded_tokens'].apply(pos_t)
        #
        # print(df_train[['paded_tokens','pos_tag_tokens']])

        # ##################################################WORD EMBEDDINGS
        # unique_words = set()
        # for tokens in list(df_train['lemma_tokens'].values):
        #     unique_words.update(tokens)
        #
        # print('Count of Unique words:', len(unique_words))
        #
        #
        #
        # word2idx = {}
        # for word in unique_words:
        #     word2idx[word] = len(word2idx)
        # word2idx['<END>'] = len(word2idx)
        #
        # word_embeddings = np.random.rand(len(word2idx), 50)
        # with open('./glove.6B/glove.6B.50d.txt', 'r',encoding='utf8') as embeds:
        #     embeddings = embeds.read()
        #     embeddings = embeddings.split('\n')[:-2]
        #
        # for token_idx, token_embed in enumerate(embeddings):
        #     token = token_embed.split()[0]
        #     if token in word2idx:
        #         word_embeddings[word2idx[token]] = [float(val) for val in token_embed.split()[1:]]
        #
        # print(f'Word embeddings for word {list(word2idx.keys())[400]}:', word_embeddings[400])

        # with open(data_file, "wb") as dill_file:
        #     dill.dump(df_train, dill_file)

        ##################################################WORD EMBEDDINGS
        unique_words = set()
        for tokens in list(df_train['lemma_tokens'].values):
            unique_words.update(tokens)

        print('Count of Unique words (training dataset):', len(unique_words))

        unique_words = set()
        for tokens in list(df_test['lemma_tokens'].values):
            unique_words.update(tokens)

        print('Count of Unique words (test dataset):', len(unique_words))

        unique_words = set()
        for tokens in list(df_train['lemma_tokens'].values)+list(df_test['lemma_tokens'].values):
            unique_words.update(tokens)

        print('Count of Unique words:', len(unique_words))


        word2idx = {}
        for word in unique_words:
            word2idx[word] = len(word2idx)
        #word2idx['<END>'] = len(word2idx)

        word_embeddings = np.random.rand(len(word2idx), 50)
        with open(additional_prefix+'../glove.6B/glove.6B.50d.txt', 'r', encoding='utf8') as embeds:
            embeddings = embeds.read()
            embeddings = embeddings.split('\n')[:-2]

        for token_idx, token_embed in enumerate(embeddings):
            token = token_embed.split()[0]
            if token in word2idx:
                word_embeddings[word2idx[token]] = [float(val) for val in token_embed.split()[1:]]

        print(f'Word embeddings for word {list(word2idx.keys())[400]}:', word_embeddings[400])

        ###############
        print(df_train[['lemma_tokens', 'sentiment']].head())
        df_train = df_train.reset_index()
        print(df_test[['lemma_tokens', 'sentiment']].head())
        df_test = df_test.reset_index()
        # df_train['embeddings'] = df_train['lemma_tokens'].apply(lambda x:
        #                                                         (np.array([word_embeddings[word2idx[word]]]) for word in x))
        # embedding_array=np.array([np.array([np.array([word_embeddings[word2idx[word]]],dtype=np.float32) for word in review]) for review in df_train['lemma_tokens']])#.squeeze()
        # df_train['embeddings'] =  embedding_array
        # # train_loader = torch.utils.data.DataLoader({1:2,3:4,5:2},shuffle=True)
        # #train_loader = torch.utils.data.DataLoader(list(zip(df_train['sentiment'].to_numpy(),embeddings)),shuffle=True)
        # train_loader = torch.utils.data.DataLoader(df_train[['sentiment','embeddings']].to_numpy(),shuffle=True)
        embedding_array = np.array(
            [np.array([np.array([word_embeddings[word2idx[word]]], dtype=d_type) for word in review]).squeeze() for
             review in
             df_train['lemma_tokens']])  # .squeeze()
        shape=list(embedding_array.shape)
        shape.insert(1,1)
        embedding_array=embedding_array.reshape(shape)#[1]+list(embedding_array.shape))
        #df_train['embeddings'] = embedding_array
        train_loader = torch.utils.data.DataLoader(list(zip(embedding_array,list(df_train['sentiment']))), shuffle=True)

        test_embedding_array = np.array(
            [np.array([np.array([word_embeddings[word2idx[word]]], dtype=d_type) for word in review]).squeeze() for
             review in
             df_test['lemma_tokens']])  # .squeeze()
        test_shape = list(test_embedding_array.shape)
        test_shape.insert(1, 1)
        test_embedding_array = test_embedding_array.reshape(test_shape)  # [1]+list(embedding_array.shape))
        #df_test['embeddings'] = test_embedding_array
        test_loader = torch.utils.data.DataLoader(list(zip(test_embedding_array,list(df_test['sentiment']))), shuffle=True)

        if cache_preprocessing:
            with open(data_file, "wb") as dill_file:
                #dill.dump(df_train, dill_file)
                dill.dump((train_loader,test_loader), dill_file)
        # #print(list(zip(df_train['embeddings'].to_numpy(), df_train['sentiment'].to_numpy())))
        # #print({k: v for k, v in zip(df_train['embeddings'].to_numpy(), df_train['sentiment'].to_numpy())})
        # # print(df_train['embeddings'].to_numpy())
        # arr=[]
        # for word in df_train['lemma_tokens'][1][0:]:
        #     x=np.array(word_embeddings[word2idx[word]])
        #     arr.append(x)
        # print(arr)
        # # print(np.array([word_embeddings[word2idx[word]] for word in df_train['lemma_tokens'][1][0:]]))
        # # df_train = df_train.reset_index()
        # # #print(list(enumerate(train_loader)))
        # # for x in enumerate(train_loader):
        # #     print(x)
    return (train_loader,test_loader)

class LambdaBase(torch.nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))
def get_model_layers(d_type=torch.float,model_nr=1):
    # if model_nr==3:
    #     layers= [torch.nn.Conv2d(1, 50, (1, 50), stride=1, padding='valid', dtype=d_type),
    #         torch.nn.Tanh(),
    #         Lambda(lambda x: x.permute(0,3,2,1)),
    #         torch.nn.Conv2d(1, 40, (1, 50), stride=1, padding='valid', dtype=d_type),
    #         torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             torch.nn.Conv2d(1, 35, (1, 40), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             torch.nn.Conv2d(1, 30, (1, 35), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             torch.nn.Conv2d(1, 27, (1, 30), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             torch.nn.Conv2d(1, 24, (1, 27), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             torch.nn.Conv2d(1, 21, (1, 24), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             torch.nn.Conv2d(1, 18, (1, 21), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             torch.nn.Conv2d(1, 16, (1, 18), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             torch.nn.Conv2d(1, 14, (1, 16), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             torch.nn.Conv2d(1, 12, (1, 14), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             torch.nn.Conv2d(1, 10, (1, 12), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             torch.nn.Conv2d(1, 8, (1, 10), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             torch.nn.Conv2d(1, 6, (1, 8), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             torch.nn.Conv2d(1, 5, (1, 6), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #
    #             torch.nn.Conv2d(1, 5, (3, 5), stride=2, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #              ]
    #     for i in range(25):
    #         layers+=(lambda: [
    #             torch.nn.Conv2d(1, 5, (3, 5), stride=1, padding='valid', dtype=d_type),
    #             torch.nn.Tanh(),
    #             Lambda(lambda x: x.permute(0, 3, 2, 1)),
    #             ])()
    #     # layers+=    [torch.nn.Flatten(),
    #     #     torch.nn.LazyLinear(2,dtype=d_type)]
    #     #41 nonlinear activations
    #     layers += [torch.nn.Flatten(),
    #                torch.nn.Linear(50,25, dtype=d_type),
    #                torch.nn.Tanh(),
    #                torch.nn.Linear(25, 13, dtype=d_type),
    #                torch.nn.Tanh(),
    #                torch.nn.Linear(13, 7, dtype=d_type),
    #                torch.nn.Tanh(),
    #                torch.nn.Linear(7, 4, dtype=d_type),
    #                torch.nn.Tanh(),
    #                torch.nn.Linear(4, 2, dtype=d_type),
    #                ]
    #     return layers
    if model_nr==3:
        layers= [torch.nn.Conv2d(1, 50, (1, 50), stride=1, padding='valid', dtype=d_type),
            torch.nn.Tanh(),
            #Lambda(lambda x: x.permute(0,3,2,1)),
            torch.nn.Conv2d(50, 40, (1, 1), stride=1, padding='valid', dtype=d_type),
            torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(40, 35, (1, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(35, 30, (1, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(30, 27, (1, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(27, 24, (1, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(24, 21, (1, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(21, 18, (1, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(18, 16, (1, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(16, 14, (1, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(14, 12, (1, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(12, 10, (1, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(10, 8, (1, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(8, 6, (1, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(6, 5, (1, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),

                torch.nn.Conv2d(5, 5, (3, 1), stride=2, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                 ]
        for i in range(25):
            layers+=(lambda: [
                torch.nn.Conv2d(5, 5, (3, 1), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                #Lambda(lambda x: x.permute(0, 3, 2, 1)),
                ])()
        # layers+=    [torch.nn.Flatten(),
        #     torch.nn.LazyLinear(2,dtype=d_type)]
        #41 nonlinear activations
        layers += [torch.nn.Flatten(),
                   torch.nn.Linear(50,25, dtype=d_type),
                   torch.nn.Tanh(),
                   torch.nn.Linear(25, 13, dtype=d_type),
                   torch.nn.Tanh(),
                   torch.nn.Linear(13, 7, dtype=d_type),
                   torch.nn.Tanh(),
                   torch.nn.Linear(7, 4, dtype=d_type),
                   torch.nn.Tanh(),
                   torch.nn.Linear(4, 2, dtype=d_type),
                   ]
        return layers
    if model_nr==2:
        layers= [torch.nn.Conv2d(1, 50, (1, 50), stride=1, padding='valid', dtype=d_type),
            torch.nn.Tanh(),
            Lambda(lambda x: x.permute(0,3,2,1)),
            torch.nn.Conv2d(1, 40, (1, 50), stride=1, padding='valid', dtype=d_type),
            torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(1, 35, (1, 40), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(1, 30, (1, 35), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(1, 27, (1, 30), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(1, 24, (1, 27), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(1, 21, (1, 24), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(1, 18, (1, 21), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(1, 16, (1, 18), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(1, 14, (1, 16), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(1, 12, (1, 14), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(1, 10, (1, 12), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(1, 8, (1, 10), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(1, 6, (1, 8), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                torch.nn.Conv2d(1, 5, (1, 6), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),

                torch.nn.Conv2d(1, 5, (3, 5), stride=2, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                 ]
        for i in range(25):
            layers+=(lambda: [
                torch.nn.Conv2d(1, 5, (3, 5), stride=1, padding='valid', dtype=d_type),
                torch.nn.Tanh(),
                Lambda(lambda x: x.permute(0, 3, 2, 1)),
                ])()
        # layers+=    [torch.nn.Flatten(),
        #     torch.nn.LazyLinear(2,dtype=d_type)]
        #41 nonlinear activations
        layers += [torch.nn.Flatten(),
                   torch.nn.Linear(50,25, dtype=d_type),
                   torch.nn.Tanh(),
                   torch.nn.Linear(25, 13, dtype=d_type),
                   torch.nn.Tanh(),
                   torch.nn.Linear(13, 7, dtype=d_type),
                   torch.nn.Tanh(),
                   torch.nn.Linear(7, 4, dtype=d_type),
                   torch.nn.Tanh(),
                   torch.nn.Linear(4, 2, dtype=d_type),
                   ]
        return layers
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
    if model_nr==1:
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

if __name__=='__main__':
    additional_prefix='../'

    d_type=torch.float32
    train_loader, test_loader=get_preprocessed_IMDB(d_type=d_type)
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, shuffle=True, batch_size=128)
    test_loader = torch.utils.data.DataLoader(test_loader.dataset, shuffle=True, batch_size=128)

    x=list(enumerate(test_loader))
    import main_linear_avggrad_implementation as main
    import torchsummary

    model = main.NN(get_model_layers(d_type=d_type,model_nr=2))
    # #print(model(torch.tensor([list(enumerate(test_loader))[0][1][1]])))
    # x=list(enumerate(test_loader))[0][1][0]
    # print([1]+list(x.shape))
    # x=x.reshape([1]+list(x.shape))
    print(np.array(x[0][1][0]).shape)
    #print(model(x))
    if d_type==torch.float:
        torchsummary.summary(model,(1,122,50))
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


