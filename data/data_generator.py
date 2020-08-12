import string
import json
import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical



class DataGenerator:
    """ Generate training/validation/test sets"""

    def __init__(self, config):
        'read parameters from config'
        FILENAME = config.filename #data filename
        OOV = config.oov #out-of-vocabulary dictionary 
        # EMBEDDING_DIM = config.model.embedding_dim # how big is each word vector
        MAX_NB_WORDS = config.model.max_nb_words # how many unique words to use (i.e num rows in embedding vector)
        MAX_SEQUENCE_LENGTH = config.model.max_sequence_length # max number of words in a question to use
         
        # read data   
        data = pd.read_csv(FILENAME, sep=';')
        data = data[["Note", "NPS", "Verbatim"]].dropna().reset_index(drop=True)
        # remove records where Note is very different from NPS 
        diff = data['Note']*2 - data['NPS']
        idx_keep = [idx for idx, val in enumerate(diff) if (val < 5) and (val > -5)] 
        data  = data.loc[idx_keep].reset_index(drop=True)
        # create labels
        data.loc[:,"Label"] = data["NPS"].apply(lambda x: self.__create_label(x))
    
        # preprocessing
        with open(OOV, "r") as f:
            dict_replace_oov = json.load(f)
        nlp = spacy.load('fr_core_news_md')
        keep_pos = ['ADV','VERB','NOUN','ADJ','PROPN','INTJ']
        data.loc[:,"Lemma"] = data["Verbatim"].apply(lambda x: self.__preprocessing_verbatim(x, dict_replace_oov, nlp, keep_pos))

        # train test split
        X = data['Lemma']
        y = data['Label']
        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=42)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
        ## fill up the missing values
        X_train = X_train.fillna("<UNK>").values
        X_test = X_test.fillna("<UNK>").values
        y_train = list(y_train.values)
        y_test = list(y_test.values)

        # tokenize the sequence
        ## Tokenize the sentences
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(list(X_train)) # creates the vocabulary index based on word frequency
        X_train_tok = tokenizer.texts_to_sequences(X_train)
        X_test_tok = tokenizer.texts_to_sequences(X_test)
        self.word_index = tokenizer.word_index
        ## Pad the sentences 
        self.X_train= pad_sequences(X_train_tok, maxlen=MAX_SEQUENCE_LENGTH)
        self.X_test = pad_sequences(X_test_tok, maxlen=MAX_SEQUENCE_LENGTH)
        y_binary = to_categorical(y_train + y_test)
        self.y_train, self.y_test = y_binary[:len(y_train)], y_binary[len(y_train):]
        
    def __create_label(self, nps):
        if nps<=6:
            return 0
        elif nps>8:
            return 2
        else:
            return 1       

    def __remove_chars(self, row):
        cust_list = [r'\d+','\n']
        del_list = string.punctuation + string.digits
        for i in del_list :
            row = row.replace(i,' ')
        for i in cust_list :
            row = row.replace(i,' ')
        return ' '.join(row.lower().split())
 
    def __replace_oov(self, row, oov):        
        return ' '.join([oov.get(v, v) for v in row.split()])
    
    def __lemma_string(self, row, nlp, keep_pos): 
        doc = nlp(row)
        return ' '.join( [token.lemma_ for token in doc if ((token.is_stop==False)|(token.pos_ in keep_pos))] )
        
    def __preprocessing_verbatim(self, row, oov, nlp, keep_pos):
        # remove useless chars
        row = self.__remove_chars(row)
        # replace oov words
        row = self.__replace_oov(row, oov)
        # tag words and keep only useful component
        row = self.__lemma_string(row, nlp, keep_pos)
        return row

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_word_index(self):
        return self.word_index