import pandas as pd
import numpy as np
from helper.reorientate import Reorientate_axis
from helper.feat_eng import Feat_eng

def main():
    
    #Read data
    part0 = pd.read_csv('part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
    part1 = pd.read_csv('part-00001-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
    part2 = pd.read_csv('part-00002-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
    part3 = pd.read_csv('part-00003-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
    part4 = pd.read_csv('part-00004-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
    part5 = pd.read_csv('part-00005-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
    part6 = pd.read_csv('part-00006-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
    part7 = pd.read_csv('part-00007-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
    part8 = pd.read_csv('part-00008-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
    part9 = pd.read_csv('part-00009-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')

    data = pd.concat([part0,part1,part2,part3,part4,part5,part6,part7,part8,part9],axis=0)

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

        #Check for at least 1 min of trip data remaining to include this trip in train data
        if ts.shape[0] < 60: continue

        #Reorientate axis to get lateral acceleration vector for lane change event
        reorientate_axis = Reorientate_axis(ts, low_pass_cutoff = 0.1)   
        reorientate_axis.reorientate(window = 100)   
        ts = reorientate_axis.ts

        #sanity checks for axis reorientation
        if ts['acc_z'].mean() < 8: continue
        if (np.max(ts['acc_z']) * np.min(ts['acc_z'])) < 0: continue

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
    preprocess.to_csv('preprocess.csv',header =True,index=False)
    
if __name__ == "__main__":
    main()