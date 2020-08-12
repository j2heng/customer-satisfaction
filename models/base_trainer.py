from sklearn.metrics import f1_score, classification_report,confusion_matrix
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
import time

class BaseTrain(object):
    def __init__(self, model, config):
        self.model_classname = model.__class__.__name__
        self.model = model.model
        self.config = config

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

class SimpleTrainer(BaseTrain):
    """simple fit"""

    def fit(self, X, y):
        self.model.fit(X, y, batch_size=self.config.batch_size, epochs=self.config.epochs,
                        validation_split=self.config.validation_split)

    def predict(self, X):
        pred = self.model.predict([X], batch_size=1024, verbose=0)
        y_pred = np.zeros_like(pred)
        y_pred[np.arange(len(pred)), pred.argmax(axis=1)] = 1
        return y_pred
        
    def score(self, X, y):
        y_pred = self.predict(X)
        score = f1_score(y, y_pred, average='macro')
        print("Test F1 Score: {:.4f}".format(score))
        print('Classification Report:\n', ' Negative(0), Neutral(1), Positive(2) \n',classification_report(y,y_pred),'\n')
        
class CBTrainer(SimpleTrainer):
    """fit with callbacks"""

    def fit(self, X, y):
        checkpoint_path = os.path.join(self.config.callbacks.checkpoint_subdir, 
                            '%s-{epoch:02d}-{val_f1_macro:.4f}.hdf5' % self.model_classname)
        checkpoints = ModelCheckpoint(checkpoint_path,
                                    monitor=self.config.callbacks.monitor, 
                                    mode=self.config.callbacks.mode,
                                    save_best_only=self.config.callbacks.save_best_only,
                                    verbose=self.config.callbacks.verbose) 
        reduce_lr = ReduceLROnPlateau(monitor=self.config.callbacks.monitor,
                                    factor=self.config.callbacks.factor,
                                    patience=self.config.callbacks.patience,
                                    verbose=self.config.callbacks.verbose,
                                    min_lr=self.config.callbacks.min_lr)
        self.model.fit(X, y, batch_size=self.config.batch_size, epochs=self.config.epochs,
                        validation_split=self.config.validation_split, callbacks=[checkpoints, reduce_lr])
