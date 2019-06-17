import pandas as pd
import numpy as np

from keras.utils import to_categorical, Sequence
from keras.models import Sequential
from keras.layers import Activation,LeakyReLU,Dense,Dropout,BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from random import shuffle
from keras.callbacks import Callback, LearningRateScheduler

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

class Model:
    '''
    Multilayer perceptron neural network to predict dangerous driving 
    '''
    def __init__(self, X_train, X_val, y_train, y_val):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = to_categorical(y_train)
        self.y_val = to_categorical(y_val)
        
        self.model = None

    def compile_model(self):
        Initializer = 'he_normal'
        activation = LeakyReLU(alpha=0.1)
        Regularizer = l2
        regparam = 1e-3

        model = Sequential()

        model.add(Dense(32,kernel_initializer=Initializer,kernel_regularizer=Regularizer(regparam),input_shape=self.X_train.shape[1:]))
        model.add(BatchNormalization(momentum=0.9))
        model.add(activation)
        model.add(Dropout(0.5,seed=42))

        model.add(Dense(64,kernel_initializer=Initializer,kernel_regularizer=Regularizer(regparam)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(activation)
        model.add(Dropout(0.5,seed=42))

        model.add(Dense(128,kernel_initializer=Initializer,kernel_regularizer=Regularizer(regparam)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(activation)
        model.add(Dropout(0.6,seed=42))

        model.add(Dense(256,kernel_initializer=Initializer,kernel_regularizer=Regularizer(regparam)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(activation)
        model.add(Dropout(0.7,seed=42))

        model.add(Dense(64,kernel_initializer=Initializer,kernel_regularizer=Regularizer(regparam)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(activation)
        model.add(Dropout(0.5,seed=42))

        model.add(Dense(16,kernel_initializer=Initializer,kernel_regularizer=Regularizer(regparam)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(activation)
        model.add(Dropout(0.5,seed=42))

        model.add(Dense(2,kernel_initializer=Initializer))
        model.add(Activation('softmax'))

        # Compile model
        adam = Adam(beta_1=0.9, beta_2=0.999, decay=0)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam)
        
        self.model = model
    
    def get_clear_pos_neg_samples(self):
        idx_pos = np.where(np.array([x[1] for x in self.y_train])==1)
        idx_neg = np.where(np.array([x[1] for x in self.y_train])==0)

        neg_mean = np.mean(self.X_train[idx_neg,:][0],axis = 1).mean()
        neg_std = np.mean(self.X_train[idx_neg,:][0],axis = 1).std()
        cutoff_pos = neg_mean + 2*neg_std
        clear_pos = np.intersect1d(np.where(np.mean(self.X_train,axis=1)>cutoff_pos)[0],idx_pos)

        pos_mean = np.mean(self.X_train[idx_pos,:][0],axis = 1).mean()
        pos_std = np.mean(self.X_train[idx_pos,:][0],axis = 1).std()
        cutoff_neg = pos_mean - 2*pos_std
        clear_neg = np.intersect1d(np.where(np.mean(self.X_train,axis=1)<cutoff_neg)[0],idx_neg)
        
        return clear_pos, clear_neg
    
    def sgd_with_restarts(self):
        
        def lr_sch(epoch):
            '''
            Learning rate schedule for stochastic gradient descent with restarts (cosine annealing)
            '''
            min_lr = 1e-5
            max_lr = 1e-3
            # number of epochs to restart
            restart = 10 

            lrate = min_lr + 0.5*(max_lr - min_lr) * (1+np.cos(((epoch - (epoch//restart * restart))/ restart) * np.pi))      

            return lrate
        
        lrsch = LearningRateScheduler(lr_sch)
        
        return lrsch
    
    def fit_model(self, clear_pos, clear_neg, lrsch, metrics):        
        batch_size = 512

        modelfit = self.model.fit_generator(balanced_generator(self.X_train,self.y_train,batch_size,clear_pos,clear_neg),
                steps_per_epoch = 50,                         
                epochs = 500,
                verbose = 0,
                validation_data = (self.X_val, self.y_val),
                callbacks = [lrsch , metrics] 
                )
        
        self.model.save("model.h5")
    
    def eval_model(self):        
        y_pred = np.array([x[1] for x in self.model.predict(self.X_train)])
        y_true = np.array([x[1] for x in self.y_train])

        print('--------------------------------------')
        print('Train:')
        print('   Precision: {0}'.format(precision_score(y_true,(y_pred>=0.5).astype(int))))
        print('   Recall: {0}'.format(recall_score(y_true,(y_pred>=0.5).astype(int))))
        print('   F1 Score: {0}'.format(f1_score(y_true,(y_pred>=0.5).astype(int))))                             
        print()
        print('   ROC-AUC Score: {0}'.format(roc_auc_score(y_true,y_pred)))
        print('   Average Precision Score: {0}'.format(average_precision_score(y_true,y_pred)))

        y_pred = np.array([x[1] for x in self.model.predict(self.X_val)])
        y_true = np.array([x[1] for x in self.y_val])

        print('--------------------------------------')
        print('Test:')
        print('   Precision: {0}'.format(precision_score(y_true,(y_pred>=0.5).astype(int))))
        print('   Recall: {0}'.format(recall_score(y_true,(y_pred>=0.5).astype(int))))
        print('   F1 Score: {0}'.format(f1_score(y_true,(y_pred>=0.5).astype(int))))                                                      
        print()
        print('   ROC-AUC Score: {0}'.format(roc_auc_score(y_true,y_pred))) 
        print('   Average Precision Score: {0}'.format(average_precision_score(y_true,y_pred)))
        print('--------------------------------------')
               
class balanced_generator(Sequence):
    '''
    Keras generator to generate a balanced mini-batch from an imbalance (1:3) train dataset 
    '''
    def __init__(self, x_set, y_set, batch_size, clear_pos, clear_neg):
        self.x, self.y = x_set, np.array([x[1] for x in y_set])
        self.batch_size = batch_size
        self.clear_pos = clear_pos
        self.clear_neg = clear_neg
           
    def __len__(self):
        #return int(np.ceil(len(self.x) / float(self.batch_size)))
        return 50

    def __getitem__(self, idx):

        rand_num = np.random.uniform(size=self.batch_size)
        
        n_neg_all = np.sum(rand_num < 0.5)
        n_clear_neg = np.sum(rand_num < 0.05)
        n_neg = n_neg_all - n_clear_neg
        
        n_pos_all = np.sum(rand_num >= 0.5)
        n_clear_pos = np.sum(rand_num > 0.95)
        n_pos = n_pos_all - n_clear_pos
        
        x_pos_idx = np.where(self.y == 1)[0]
        x_clear_pos_idx = self.clear_pos 
        x_neg_idx = np.where(self.y == 0)[0]
        x_clear_neg_idx = self.clear_neg
        
        sel_pos_idx = np.random.choice(x_pos_idx,n_pos, replace = False)
        sel_clear_pos_idx = np.random.choice(x_clear_pos_idx,n_clear_pos, replace= False)
        sel_neg_idx = np.random.choice(x_neg_idx,n_neg, replace = False)
        sel_clear_neg_idx = np.random.choice(x_clear_neg_idx,n_clear_neg, replace= False)
        
        sel_idx = np.concatenate([sel_pos_idx, sel_clear_pos_idx, sel_neg_idx, sel_clear_neg_idx])
        sel_label = [1]*(n_pos+n_clear_pos) + [0]*(n_neg+n_clear_neg)
        
        sel_idx_label = list(zip(sel_idx,sel_label))
        shuffle(sel_idx_label)
        
        sel_idx = np.array([x[0] for x in sel_idx_label])
        sel_label = np.array([x[1] for x in sel_idx_label])

        x_train_resample = self.x[sel_idx]
        y_train_resample = to_categorical(sel_label)

        return x_train_resample, y_train_resample
    
class Metrics(Callback):
    '''
    Print validation ROC-AUC score and Average Precision score after every epoch trained  
    '''
    
    def on_epoch_end(self, epoch, logs={}):
        val_prob = np.array([x[1] for x in (np.asarray(self.model.predict(self.validation_data[0])))]).reshape(-1, 1) 
        val_targ = np.array([x[1] for x in (np.asarray(self.validation_data[1]))]).reshape(-1, 1) 
        val_roc_auc = roc_auc_score(val_targ, val_prob)
        ave_prec = average_precision_score(val_targ, val_prob)

        if (epoch+1)%20 == 0: 
            print ("Epoch {0}: Val. ROC-AUC Score = {1}, Val. Average Precision Score = {2}".format(epoch+1, np.round(val_roc_auc,3), np.round(ave_prec,3)))
        return
    
class Model_full(Model):
    '''
    Used in train.py to retrain model over the full dataset  
    '''
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = to_categorical(y_train)
        
        self.model = None
        
        
    def fit_model(self, clear_pos, clear_neg, lrsch):        
        batch_size = 512

        modelfit = self.model.fit_generator(balanced_generator(self.X_train,self.y_train,batch_size,clear_pos,clear_neg),
                steps_per_epoch = 50,                         
                epochs = 500,
                verbose = 0,
                callbacks = [lrsch] 
                )
        
        self.model.save("model_full.h5")
        
    
        
