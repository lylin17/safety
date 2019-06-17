import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

'''
Utility functions used in various notebook and scripts
'''

def collinearity_check(preprocess):
    '''
    Plot correlation matrix as a heatmap to check correlations between features in solution.ipynb
    '''
    features = preprocess.iloc[:,1:].columns

    f,ax = plt.subplots(1,1,figsize=(15, 15))

    im = ax.matshow(preprocess.iloc[:,1:].corr(),interpolation='none')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    f.colorbar(im,cax=cax)

    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(features)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features)

    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
             ha="left", va="center",rotation_mode="anchor")

    plt.show()
    
def train_val_split(preprocess, labels):
    '''
    Merge features to labels by bookingID and perform 80:20 train-validation split in solution.ipynb
    '''
    train_labelsID = [x for x in np.unique(labels['bookingID']) if x in np.unique(preprocess['bookingID'])]

    train_labels = labels[labels['bookingID'].isin(train_labelsID)]
    train_labels = train_labels.drop_duplicates(subset = 'bookingID',keep=False)

    data = preprocess.merge(train_labels,how='inner',on='bookingID')
    
    X_train, X_val, y_train, y_val = train_test_split(data.iloc[:,1:-1], np.array(data.iloc[:,-1]),train_size = 0.8, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val

def normalize(X_train, X_val):
    '''
    Normalize features for input to neural network model in solution.ipynb
    '''
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    
    np.save('data_min.npy',np.array(scaler.data_min_))
    np.save('data_range.npy', np.array(scaler.data_range_))
    
    return X_train, X_val

def normalize_full(X_train):
    '''
    Normalize features for full dataset using min and range of train data in train.py and test.py
    '''
    data_min = np.load('data_min.npy')
    data_range = np.load('data_range.npy')
    
    X_train = X_train - np.broadcast_to(data_min, X_train.shape)
    X_train = X_train /  np.broadcast_to(data_range, X_train.shape)
    
    return X_train

def output_test_pred(model, X_test, ids):
    '''
    output predicted probabilities for hold-out set in test.py
    '''
    y_pred = np.array([x[1] for x in model.predict(X_test)])
    
    pred_df = pd.DataFrame([ids,y_pred])
    pred_df = pred_df.T
    pred_df.columns = ['bookingID','Predicted Probabilities']
    pred_df.to_csv('prediction.csv', header =True, index=False)

def output_test_eval(model, X_test, y_test):
    '''
    output evalution metric scores for hold-out set in test.py
    '''    
    y_pred = np.array([x[1] for x in model.predict(X_test)])
    y_true = np.array(y_test)
        
    file = open("evaluation.txt","w")

    file.write('Model Evaluation on Hold-out Set\n')
    file.write('--------------------------------------\n')
    file.write('   Precision: {0}\n'.format(np.round(precision_score(y_true,(y_pred>=0.5).astype(int)),3)))
    file.write('   Recall: {0}\n'.format(np.round(recall_score(y_true,(y_pred>=0.5).astype(int)),3)))
    file.write('   F1 Score: {0}\n'.format(np.round(f1_score(y_true,(y_pred>=0.5).astype(int)),3)))                             
    file.write('\n')
    file.write('   ROC-AUC Score: {0}\n'.format(np.round(roc_auc_score(y_true,y_pred),3)))
    file.write('   Average Precision Score: {0}\n'.format(np.round(average_precision_score(y_true,y_pred),3)))
    file.write('--------------------------------------')
    
    file.close()

