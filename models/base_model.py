import spacy
import numpy as np
from keras.layers import Embedding, Input, Bidirectional, GRU, Dropout, Dense, LSTM
from keras.models import Model
from utils.metrics import f1_macro, f1_micro
from models.Attention import Attention

class BaseModel:
    def __init__(self, config, word_index):
        self.config = config
        self.word_index = word_index
        self.model = None
        self.build_model()

    def generate_embedding_matrix(self):
        nlp = spacy.load('fr_core_news_md')
        self.NB_WORDS = min(self.config.model.max_nb_words, len(self.word_index) + 1)
        embedding_matrix = np.zeros((self.NB_WORDS, self.config.model.embedding_dim))
        for word, i in self.word_index.items():
            if i >= self.NB_WORDS: continue 
            token = nlp(word)
            embedding_vector = token.vector
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("No model to save.")
        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("No model to load.")
        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self):
        raise NotImplementedError

class BiGRUAttention(BaseModel):
    """Single-layer Bidirectional GRU with Attention"""
    def build_model(self):
        embedding_matrix = self.generate_embedding_matrix()
        inp = Input(shape=(self.config.model.max_sequence_length,))
        x = Embedding(self.NB_WORDS, self.config.model.embedding_dim, weights=[embedding_matrix])(inp)
        x = Bidirectional(GRU(self.config.model.BiGRUAttention.gru_units, return_sequences=True))(x)
        x = Attention(self.config.model.max_sequence_length)(x) # New
        x = Dense(self.config.model.BiGRUAttention.dense_1, activation=self.config.model.BiGRUAttention.dense_activation_1)(x)
        x = Dropout(self.config.model.BiGRUAttention.dropout)(x)
        x = Dense(3, activation='softmax')(x)
        self.model = Model(inputs=inp, outputs=x)
        self.model.compile(loss=self.config.compile.loss, optimizer=self.config.compile.optimizer, metrics=['accuracy',f1_macro])
        print(self.model.summary())
            
class BiLSTMAttention(BaseModel):
    """Two-layer Bidirectional LSTM with Attention"""
    def build_model(self):
        embedding_matrix = self.generate_embedding_matrix()
        inp = Input(shape=(self.config.model.max_sequence_length,))
        x = Embedding(self.NB_WORDS, self.config.model.embedding_dim, weights=[embedding_matrix], trainable=False)(inp)
        x = Bidirectional(LSTM(self.config.model.BiLSTMAttention.lstm_units_1, return_sequences=True))(x)
        x = Bidirectional(LSTM(self.config.model.BiLSTMAttention.lstm_units_2, return_sequences=True))(x)
        x = Attention(self.config.model.max_sequence_length)(x)
        x = Dense(self.config.model.BiLSTMAttention.dense_1, activation=self.config.model.BiLSTMAttention.dense_activation_1)(x)
        x = Dense(3, activation='softmax')(x)
        self.model = Model(inputs=inp, outputs=x)
        self.model.compile(loss=self.config.compile.loss, optimizer=self.config.compile.optimizer, metrics=['accuracy',f1_macro])
        print(self.model.summary())