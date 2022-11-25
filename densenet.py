import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import re 
import time
import os
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold,train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.layers import Input,add,Dense,concatenate,Flatten,Conv1D,AveragePooling1D,Dropout,Reshape,multiply,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal,GlorotUniform
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from dataclasses import dataclass
from typing import Optional
from io import TextIOBase
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def repeat_code(dir_out,num_epoch):
    def dense_layer(x,filters = 12,k = 1,regular_rate = 1e-4,drop_rate = 0.2):
      #x = BatchNormalization(epsilon=1.001e-5)(x)
      x = Conv1D(filters = filters*k,kernel_size = 1,strides = 1,padding = 'same',activation = 'relu',
                kernel_initializer = 'glorot_normal', kernel_regularizer = l2(regular_rate))(x)
      x = Conv1D(filters = filters,kernel_size = 3,strides = 1,padding = 'same',activation = 'relu',
                kernel_initializer = 'glorot_normal', kernel_regularizer = l2(regular_rate))(x)
      x = Dropout(drop_rate)(x)
      return x

    def dense_block(x,layers = 4,filters = 12,k =1,regular_rate = 1e-4,drop_rate = 0.2):
      for i in range(layers):
          conv = dense_layer(x,filters = filters,k = k,regular_rate = regular_rate,drop_rate=drop_rate)
          x = concatenate([x, conv], axis = 2)    
      return x

    def transition_layer(x,compression_rate = 0.5,regular_rate = 1e-4):
      filters = int(x.shape.as_list()[-1]*compression_rate)
      x = Conv1D(filters = filters,kernel_size = 1,strides = 1,padding = 'same',activation = 'relu',
                  kernel_initializer = 'glorot_normal', kernel_regularizer = l2(regular_rate))(x)
      x = AveragePooling1D(pool_size = 2)(x)
      return x
    
    def dense_model(input_shape = (500, 4),regular_rate = 1e-4):
      inputs = Input(input_shape)
      x = Conv1D(filters = 64,kernel_size = 3,strides = 1,padding = 'same',activation = 'relu',
                  kernel_initializer = 'glorot_normal', kernel_regularizer = l2(regular_rate))(inputs)
      x = Conv1D(filters = 64,kernel_size = 3,strides = 1,padding = 'same',activation = 'relu',
                  kernel_initializer = 'glorot_normal', kernel_regularizer = l2(regular_rate))(x)
      x = AveragePooling1D(pool_size = 2)(x)
      x = dense_block(x,layers =6 ,filters=12,k = 2)
      x = transition_layer(x)
      x = dense_block(x,layers = 12,filters=12,k = 4)
      x = transition_layer(x)
      x = dense_block(x,layers = 24,filters=12,k = 4)
      x = transition_layer(x)
      x = dense_block(x,layers = 16,filters=12,k = 4)
      x = Flatten()(x)
      x = Dropout(0.2)(x)
      outputs = Dense(1,activation = 'sigmoid',kernel_initializer = 'glorot_normal',kernel_regularizer = l2(regular_rate))(x)
      model = Model(inputs = inputs, outputs = outputs)
      return model

    def One_hot(dirs):
        files = open(dirs, 'r') 
        sample = []
        for line in files:
            if re.match('>', line) is None:
                value = np.zeros((500, 4), dtype='float32')
                for index, base in enumerate(line.strip()):
                    if re.match(base,'A|a'):
                        value[index,0] = 1
                    if re.match(base, 'C|c'):
                        value[index, 1] = 1
                    if re.match(base, 'G|g'):
                        value[index, 2] = 1
                    if re.match(base, 'T|t'):
                        value[index, 3] = 1
                   
                sample.append(value)
        files.close()
        return np.array(sample)


    def matrix(y_test_all, y_pred_all):
        y_test_all = np.array(y_test_all)
        y_pred_all = np.array(y_pred_all)

        auc = roc_auc_score(y_test_all, y_pred_all)

        y_pred_all[y_pred_all >= 0.5] = 1
        y_pred_all[y_pred_all < 0.5] = 0
        y_test_all.ravel()
        y_pred_all.ravel()
        acc = accuracy_score(y_test_all, y_pred_all)
        return acc, auc

    neg_background = np.load('/home/hlcheng/corn/sample/neg_sample.npy')
  
    for root, dirs, files in os.walk('/home/hlcheng/corn/get_pos_fa'): 
        for file_n in files:
            if not os.path.exists(dir_out +'/'+file_n):
                os.mkdir(dir_out +'/'+file_n)
                dir_re=os.path.join(dir_out,file_n)
                start_t = time.time()
                dir_pos_file = os.path.join(root, file_n)
                
                count=-1
                for count,line in enumerate(open(dir_pos_file,'rU').readlines()):
                    count += 1
                pos_num=count
               
                pos_matrix = One_hot(dirs=dir_pos_file)
                pos_label = np.ones([pos_matrix.shape[0], 1])
            
             
                neg_matrix_index = np.random.choice(neg_background.shape[0], pos_num,replace=False)
                neg_matrix = neg_background[neg_matrix_index]
                neg_label = np.zeros([pos_matrix.shape[0], 1])
                
               
                input_x = np.row_stack((pos_matrix, neg_matrix))
                input_y = np.row_stack((pos_label, neg_label))
                print(input_x.shape, input_y.shape)
            

               

                out = open(dir_re + '/result.txt', 'w')
                out.write('The size of total dataset is: %s\n' % input_x.shape[0])
                out.write('The size of training dataset is: %s\n' % pos_num)
            
                X_train, X_test, Y_train, Y_test = train_test_split(input_x,
                                                                    input_y,
                                                                    test_size=0.2,
                                                                    shuffle=True,
                                                                    random_state=20)
                
                print('#########################################################################')
                #print("%s repeat done!!!"% rep)
                print('#########################################################################')
                
                model = dense_model()

                checkpoint = ModelCheckpoint(filepath=dir_re+ '/checkmodel.hdf5',
                                            verbose=0,
                                            save_best_only=True,
                                            monitor='val_loss',
                                            mode='min')
                early_stopping =EarlyStopping(monitor='val_loss', patience=5)
                model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001),metrics=['accuracy'])
                model.fit(X_train,
                        Y_train,
                        batch_size=128,
                        epochs=num_epoch,
                        shuffle=True,
                        validation_split=0.1,
                        callbacks=[early_stopping,checkpoint]) 
                    
                Y_pred = model.predict(X_test)
                Y_pred_train=model.predict(X_train)
                print(Y_test.shape,Y_pred.shape)
                print(Y_pred[1])
                
                model.save(dir_re+'/checkmodel.hdf5')
                df=DataFrame()
                df['label']=Series(Y_test.flatten())
                df['pred']=Series(Y_pred.flatten())
                df.to_csv(dir_re+'/roc.csv')
                train_acc,train_auc=matrix(Y_train,Y_pred_train)
                test_acc, test_auc = matrix(Y_test, Y_pred)
                out.write('\ttrain_acc\ttrain_auc\n')
                out.write('\t%s\t%s\n'% (train_acc, train_auc))
                out.write('\ttest_acc\ttest_auc\n')
                out.write('\t%s\t%s\n' % (test_acc, test_auc))
                end_t = time.time()
                all_time = end_t - start_t
                out.write('\tusing time: %s\n' % all_time)
                out.close()
if __name__ == "__main__":         
    dir_out = '/home/hlcheng/corn/desenet_without_parameter/64_64'
    num_epoch = 80
    repeat_code(dir_out,num_epoch)