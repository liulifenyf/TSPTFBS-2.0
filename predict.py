#%%
'''
all_models_path:the path of all models (./DenseNet_models)
fasta_path: the path of sequences
wpath: the path of results (./results.csv)

'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import os
import re
import numpy as np
import pandas as pd
import sys
gpus=tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        logical_gpus=tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus),"Physical GPUs",len(logical_gpus),"Logical GPUs")
    except RuntimeError as e:
        print(e)

def get_all_models(all_models_path):
    models=[]
    for root ,dirs, files in os.walk(all_models_path):
        for f in files:
            model_path=os.path.join(root,f)
            models.append(model_path)
    return models
def One_ont(dirs):
    files=open(dirs,'r')
    sample=[]
    for line in files:
        if re.match('>',line) is None:
            value=np.zeros((500,4),dtype='float32')
            for index,base in enumerate(line.strip()):
                if re.match(base,'A|a'):
                    value[index,0]=1
                if re.match(base,'C|c'):
                    value[index,1]=1
                if re.match(base,'G|g'):
                    value[index,2]=1
                if re.match(base,'T|t'):
                    value[index,3]=1
            sample.append(value)
    files.close()
    return np.array(sample)

def predict_fasta(all_models_path,fasta_path,wpath):
    sample=One_ont(fasta_path)
    result={}
    count=0
    for one_model_path in get_all_models(all_models_path):
        model_tf=re.findall('.*/(.*)_',one_model_path)[-1]
        count+=1
        print('Now loading the trained model of factor {} for predicting! this is the num {} model, totally 389 models!'.format(model_tf, count))
        model=load_model(one_model_path)
        pred=model.predict(sample)
        result[model_tf]=list(pred[:,0])
        result_df=pd.DataFrame(result).T
        result_df.to_csv(wpath,sep=',')
    return result_df
def main():
    all_models_path='./DenseNet_models/Oryza_sativa_models'
    fasta_path=sys.argv[1]
    wpath='./results.csv'
    result_df=predict_fasta(all_models_path,fasta_path,wpath)

if __name__ =='__main__':
    main()

    
    
    


#%%
import numpy as np
a=np.array([0.76614434],[0.8909822 ])
# %%
