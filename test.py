import pandas as pd
import numpy as np

from helper.reorientate import Reorientate_axis
from helper.feat_eng import Feat_eng
from helper.utils import normalize_full, output_test_pred, output_test_eval

from keras.models import load_model

def main():
    
    #Read data
    data = pd.read_csv('test_features.csv')
    len_features = pd.read_csv('preprocess.csv',nrows=0).shape[1]

    #Initialize preprocessing
    unique_id = np.unique(data['bookingID'])   
    preprocess = pd.DataFrame(None)

    #Preprocess Data
    for id_sel in unique_id:
        ts = data.loc[data['bookingID'] == id_sel,:].sort_values(['second'])

        # Drop rows with missing speed and low GPS accuracy
        ts = ts.loc[ts['Speed'] != -1,:]
        ts = ts.loc[ts['Accuracy'] <= 20,:].reset_index(drop=True)

        # Get forward acceleration from speed, bearing change, and acceleration change
        # Any break in time sequence > 5s is considered a time jump in time series instead of missing value

        ts['forw_acc'] = 0
        ts['forw_acc_change'] = 0
        ts['Bearing_change'] = 0
        
        for i in range(ts.shape[0]-1):
            if (ts.loc[i+1,'second'] - ts.loc[i,'second']) < 5 :
                ts.loc[i,'forw_acc'] = (ts.loc[i,'Speed'] - ts.loc[i+1,'Speed'])/(ts.loc[i,'second'] - ts.loc[i+1,'second'])
                ts.loc[i,'forw_acc_change'] = (ts.loc[i,'forw_acc'] - ts.loc[i+1,'forw_acc'])/(ts.loc[i,'second'] - ts.loc[i+1,'second'])
                                
                #Ensure bearing change is between 0 and 180 degrees
                angle_change = abs(round(np.degrees(np.arctan2(np.sin(np.radians(ts.loc[i,'Bearing'])-np.radians(ts.loc[i+1,'Bearing'])),np.cos(np.radians(ts.loc[i,'Bearing'])-np.radians(ts.loc[i+1,'Bearing'])))),2))
                ts.loc[i,'Bearing_change'] = angle_change/abs(ts.loc[i,'second'] - ts.loc[i+1,'second'])

        #Check for at least 1 min of trip data remaining to include this trip in train data, if fail append all zero data
        if ts.shape[0] < 60:
            preprocess.loc[-1] = [id_sel]+ [np.nan]*(len_features-1)
            preprocess.index = preprocess.index + 1  
            continue

        #Reorientate axis to get lateral acceleration vector for lane change event
        reorientate_axis = Reorientate_axis(ts, low_pass_cutoff = 0.1)   
        reorientate_axis.reorientate(window = 100)   
        ts = reorientate_axis.ts

        #sanity checks for axis reorientation, if fail append all zero data
        if ts['acc_z'].mean() < 8: 
            preprocess.loc[-1] = [id_sel]+ [np.nan]*(len_features-1)
            preprocess.index = preprocess.index + 1
            continue
            
        if (np.max(ts['acc_z']) * np.min(ts['acc_z'])) < 0: 
            preprocess.loc[-1] = [id_sel]+ [np.nan]*(len_features-1)
            preprocess.index = preprocess.index + 1
            continue

        #Feature Engineering    
        feat_eng = Feat_eng(ts)
        
        feat_eng.speed()
        feat_eng.lr_turn()
        feat_eng.u_turn()
        feat_eng.veh_jerk()
        
        ts_full = feat_eng.full_ts()
        feat_eng.forw(ts_full)
        feat_eng.brake(ts_full)
        feat_eng.turn_acc(ts_full)
        feat_eng.lane_change(ts_full) 

        trip_features = feat_eng.output_features()

        #Add preprocessed trip features to preprocessed data
        preprocess = pd.concat([preprocess,trip_features],axis=0)
    
    # Clean up and output preprocessed data

    #Fill nan values(no event detected) with mean of the feature column disregarding the nan values
    for i in range(preprocess.shape[1]):        
        preprocess.iloc[:,i] = preprocess.iloc[:,i].fillna(np.nanmedian(preprocess.iloc[:,i]))
        
    preprocess = preprocess.reset_index(drop=True)
    preprocess.to_csv('preprocess_test.csv',header =True,index=False)    
    
    ### If for any reason the script stopped after this point, comment out the code above this point in main() to start again from this point ###
    
    #Merge features with labels
    
    preprocess = pd.read_csv('preprocess_test.csv')
    test_labels = pd.read_csv('test_labels.csv')
    test_labelsID = [x for x in np.unique(test_labels['bookingID']) if x in np.unique(preprocess['bookingID'])]

    test_labels = test_labels[test_labels['bookingID'].isin(test_labelsID)]
    test_labels = test_labels.drop_duplicates(subset = 'bookingID',keep=False)

    data = preprocess.merge(test_labels,how='inner',on='bookingID')
    
    #Normalized preprocessed data
    X_test = normalize_full(data.iloc[:,1:-1])
    y_test = np.array(data.iloc[:,-1])
    
    #Choose only the selected feature set
    feat_sel = np.load('feat_sel.npy')
    X_test = X_test.loc[:,feat_sel]
    X_test = X_test.values
 
    #Load trained model
    trained_model = load_model('model_full.h5')
    
    #Output predicted probabilities
    ids = np.array(data['bookingID'])
    output_test_pred(trained_model, X_test, ids)
    
    #Output model evalution
    output_test_eval(trained_model, X_test, y_test)
    
if __name__ == "__main__":
    main()