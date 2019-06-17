import pandas as pd
import numpy as np

from helper.utils import normalize_full
from helper.model import Model_full

def main():
    
    #Read data
    preprocess = pd.read_csv('preprocess.csv')

    #Data Preparation for Model
    
    #Merge features with labels
    labels = pd.read_csv('part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv')
    train_labelsID = [x for x in np.unique(labels['bookingID']) if x in np.unique(preprocess['bookingID'])]

    train_labels = labels[labels['bookingID'].isin(train_labelsID)]
    train_labels = train_labels.drop_duplicates(subset = 'bookingID',keep=False)

    data = preprocess.merge(train_labels,how='inner',on='bookingID')
    
    #Data Normalization
    X_train = normalize_full(data.iloc[:,1:-1])
    y_train = np.array(data.iloc[:,-1])
    
    #Choose only the selected feature set
    feat_sel = np.load('feat_sel.npy')
    X_train = X_train.loc[:,feat_sel]
    X_train = X_train.values
    
    #Initialize model
    model = Model_full(X_train, y_train)

    #Compile model
    model.compile_model()

    #Prepare for model
    clear_pos, clear_neg = model.get_clear_pos_neg_samples()
    lrsch = model.sgd_with_restarts()

    #Fit Model
    model.fit_model(clear_pos, clear_neg, lrsch)
    
if __name__ == "__main__":
    main()