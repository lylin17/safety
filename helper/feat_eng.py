import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks , peak_widths

class Feat_eng:

    def __init__(self,ts):
        
        self.ts = ts
        
        self.feat_set = np.array([np.unique(ts['bookingID'])[0]])
        self.feat_names = ['bookingID']
    
    def speed(self):
        mean_speed = np.mean(self.ts['Speed']) 
        max_speed = np.max(self.ts['Speed'])
        std_speed = np.std(self.ts['Speed'])

        if std_speed == 0: pad_speed = 0
        else: pad_speed =(max_speed-mean_speed)/std_speed

        speeding_freq = self.ts['Speed'][self.ts['Speed']>25].shape[0]/self.ts.shape[0]
        reckless_speeding_freq = self.ts['Speed'][self.ts['Speed']>30].shape[0]/self.ts.shape[0]
        
        self.feat_set = np.concatenate([self.feat_set, np.array([max_speed, pad_speed, speeding_freq, reckless_speeding_freq])])
        self.feat_names += ['max_speed', 'pad_speed','speeding_freq', 'reckless_speeding_freq']
    
    def lr_turn(self):
        lr_turn_event =  np.array(self.ts.loc[((self.ts['Bearing_change']>30) & (self.ts['Bearing_change']<150)),'second'])
        
        if len(lr_turn_event) == 0:
            mean_lr_turn_agg = np.nan 
            max_lr_turn_agg = np.nan 
            pad_lr_turn_agg = np.nan 

            mean_lr_turn_speed = np.nan  
            max_lr_turn_speed = np.nan 
            pad_lr_turn_speed = np.nan 

            harsh_turn_agg_count = 0
            harsh_turn_speed_count = 0

        else:
            #Turn aggression = speed * sqrt(Bearing change)
            #Bearing change between 30 - 150 degrees and accepted turn speed generally ~ 10m/s 
            lr_turn_agg = self.ts.loc[self.ts['second'].isin(lr_turn_event), 'Speed'] * np.sqrt(self.ts.loc[self.ts['second'].isin(lr_turn_event), 'Bearing_change']) 
            mean_lr_turn_agg = np.mean(lr_turn_agg) 
            max_lr_turn_agg = np.max(lr_turn_agg)
            std_lr_turn_agg = np.std(lr_turn_agg)

            if std_lr_turn_agg == 0: pad_lr_turn_agg = 0        
            else: pad_lr_turn_agg = (max_lr_turn_agg-mean_lr_turn_agg)/std_lr_turn_agg

            lr_turn_speed = self.ts.loc[self.ts['second'].isin(lr_turn_event), 'Speed']
            mean_lr_turn_speed = np.mean(lr_turn_speed) 
            max_lr_turn_speed = np.max(lr_turn_speed)
            std_lr_turn_speed = np.std(lr_turn_speed)

            if std_lr_turn_speed == 0: pad_lr_turn_speed = 0        
            else: pad_lr_turn_speed = (max_lr_turn_speed - mean_lr_turn_speed)/std_lr_turn_speed

            harsh_turn_agg_count = lr_turn_agg[lr_turn_agg>60].shape[0]
            harsh_turn_speed_count = lr_turn_speed[lr_turn_speed>12].shape[0]
            
        self.feat_set = np.concatenate([self.feat_set, np.array([max_lr_turn_speed, pad_lr_turn_speed, harsh_turn_speed_count, max_lr_turn_agg, pad_lr_turn_agg, harsh_turn_agg_count])])
        self.feat_names += ['max_lr_turn_speed', 'pad_lr_turn_speed', 'harsh_turn_speed_count','max_lr_turn_agg', 'pad_lr_turn_agg', 'harsh_turn_agg_count']    
    
    def u_turn(self):
        u_turn_event =  np.array(self.ts.loc[(self.ts['Bearing_change']>150) ,'second'])

        if len(u_turn_event) == 0:
            mean_u_turn_speed = np.nan 
            max_u_turn_speed = np.nan 
            pad_u_turn_speed = np.nan 

        else:
            u_turn_speed = self.ts.loc[self.ts['second'].isin(u_turn_event), 'Speed']      
            mean_u_turn_speed = np.mean(u_turn_speed) 
            max_u_turn_speed = np.max(u_turn_speed)
            std_u_turn_speed = np.std(u_turn_speed)

            if std_u_turn_speed == 0: pad_u_turn_speed = 0
            else: pad_u_turn_speed = (max_u_turn_speed-mean_u_turn_speed)/std_u_turn_speed
                
        self.feat_set = np.concatenate([self.feat_set, np.array([max_u_turn_speed, pad_u_turn_speed])])
        self.feat_names += ['max_u_turn_speed', 'pad_u_turn_speed']

    def veh_jerk(self):
        veh_jerk = abs(self.ts['forw_acc_change'])

        mean_veh_jerk = np.mean(veh_jerk)
        max_veh_jerk = np.max(veh_jerk) 
        std_veh_jerk = np.std(veh_jerk)

        if std_veh_jerk == 0: pad_veh_jerk = 0
        else: pad_veh_jerk = (max_veh_jerk-mean_veh_jerk)/std_veh_jerk

        hard_veh_jerk_count = veh_jerk[veh_jerk>5].shape[0]
        
        self.feat_set = np.concatenate([self.feat_set, np.array([max_veh_jerk, pad_veh_jerk, hard_veh_jerk_count])])
        self.feat_names += ['max_veh_jerk','pad_veh_jerk','hard_veh_jerk_count']
        
    def full_ts(self):

        #Add in missing time period in time series ts_full for accurate peak detection
        full_duration = pd.DataFrame(list(range(int(np.min(self.ts['second'])),int(np.max(self.ts['second'])))),columns = ['second'])
        ts_full = full_duration.merge(self.ts,how = 'outer', on = 'second')

        #Fill missing values with 0, will be ignored in peak detection
        ts_full = ts_full.fillna(0)

        #Smooth time series ts_full
        fs = 1
        nyquist = fs / 2
        cutoff = 0.1 * nyquist
        b, a = butter(5, cutoff, btype='lowpass')
        ts_full['acc_x'] = filtfilt(b, a, ts_full['acc_x'])
        ts_full['acc_y'] = filtfilt(b, a, ts_full['acc_y'])
        
        return ts_full
    
    def forw(self, ts_full):        
        forw_peaks, _ = find_peaks(ts_full['acc_x'],height = 0.33)
        
        if len(forw_peaks) == 0: 
            mean_forw = np.nan 
            max_forw = np.nan 
            pad_forw = np.nan 

            mean_forw_peaks = np.nan 
            max_forw_peaks = np.nan 
            pad_forw_peaks = np.nan 

            hard_forw_count = 0
            abrupt_forw_count = 0

        else:    
            forw_peaks_val = np.array(ts_full['acc_x'])[forw_peaks]
            forw_width = peak_widths(ts_full['acc_x'], forw_peaks)[0]
            forw = forw_peaks_val/forw_width

            mean_forw = np.mean(forw) 
            max_forw = np.max(forw)
            std_forw = np.std(forw)

            if std_forw == 0: pad_forw = 0
            else: pad_forw = (max_forw - mean_forw)/std_forw

            mean_forw_peaks = np.mean(forw_peaks_val)
            max_forw_peaks = np.max(forw_peaks_val)
            std_forw_peaks = np.std(forw_peaks_val)

            if std_forw_peaks == 0: pad_forw_peaks = 0
            else: pad_forw_peaks = (max_forw_peaks - mean_forw_peaks)/std_forw_peaks

            hard_forw_count = np.sum((np.array(ts_full['acc_x'])[forw_peaks])>1.35)
            abrupt_forw_count = np.sum(forw_width < 10)
            
        self.feat_set = np.concatenate([self.feat_set, np.array([max_forw_peaks, pad_forw_peaks, max_forw, pad_forw, hard_forw_count, abrupt_forw_count])])
        self.feat_names += ['max_forw_peaks', 'pad_forw_peaks', 'max_forw', 'pad_forw','hard_forw_count', 'abrupt_forw_count']

    def brake(self, ts_full):
        brake_peaks, _ = find_peaks(-ts_full['acc_x'],height = 0.33)
        
        if len(brake_peaks) == 0: 
            mean_brake = np.nan 
            max_brake = np.nan 
            pad_brake = np.nan 

            mean_brake_peaks = np.nan 
            max_brake_peaks = np.nan 
            pad_brake_peaks = np.nan 

            hard_brake_count = 0
            abrupt_brake_count = 0

        else:
            brake_peaks_val = np.array(-ts_full['acc_x'])[brake_peaks]
            brake_width = peak_widths(-ts_full['acc_x'], brake_peaks)[0]
            brake = brake_peaks_val/brake_width

            mean_brake = np.mean(brake) 
            max_brake = np.max(brake)
            std_brake = np.std(brake)

            if std_brake == 0: pad_brake = 0
            else: pad_brake = (max_brake-mean_brake)/std_brake

            mean_brake_peaks = np.mean(brake_peaks_val)
            max_brake_peaks = np.max(brake_peaks_val)
            std_brake_peaks = np.std(brake_peaks_val)

            if std_brake_peaks == 0: pad_brake_peaks = 0
            else: pad_brake_peaks = (max_brake_peaks - mean_brake_peaks)/std_brake_peaks

            hard_brake_count = np.sum((np.array(-ts_full['acc_x'])[brake_peaks])>1.35)
            abrupt_brake_count = np.sum(brake_width < 10)
            
        self.feat_set = np.concatenate([self.feat_set, np.array([max_brake_peaks, pad_brake_peaks, max_brake, pad_brake, hard_brake_count, abrupt_brake_count])])
        self.feat_names += ['max_brake_peaks', 'pad_brake_peaks', 'max_brake', 'pad_brake', 'hard_brake_count', 'abrupt_brake_count'] 

    def turn_acc(self, ts_full):
        turn_window = 20
        turn_event =  np.array(self.ts.loc[(self.ts['Bearing_change']>30) ,'second'])
        turn_event_bound = []
        for i in turn_event:
            turn_event_bound+=(list(range(int(i-turn_window),int(i+turn_window))))
        turn_event_bound = list(set(turn_event_bound))

        ts_turn = ts_full.copy(deep = True)
        ts_turn.loc[~ts_turn['second'].isin(turn_event_bound),'acc_y'] = 0

        turn_acc_peaks, _ = find_peaks(abs(ts_turn['acc_y']) ,height = 0.33, width = 5)
        if len(turn_acc_peaks) == 0:
            mean_turn_acc = np.nan 
            max_turn_acc = np.nan 
            pad_turn_acc = np.nan 

            mean_turn_acc_peaks = np.nan 
            max_turn_acc_peaks = np.nan 
            pad_turn_acc_peaks = np.nan 

            hard_turn_acc_count = 0
            abrupt_turn_acc_count = 0

        else:
            turn_acc_peaks_val = np.array(abs(ts_turn['acc_y']))[turn_acc_peaks]
            turn_acc_width = peak_widths(abs(ts_turn['acc_y']) , turn_acc_peaks)[0]
            turn_acc = turn_acc_peaks_val/turn_acc_width

            mean_turn_acc = np.mean(turn_acc) 
            max_turn_acc = np.max(turn_acc)
            std_turn_acc = np.std(turn_acc)

            if std_turn_acc == 0: pad_turn_acc = 0
            else: pad_turn_acc = (max_turn_acc - mean_turn_acc)/std_turn_acc

            mean_turn_acc_peaks = np.mean(turn_acc_peaks_val) 
            max_turn_acc_peaks = np.max(turn_acc_peaks_val)
            std_turn_acc_peaks = np.std(turn_acc_peaks_val)

            if std_turn_acc_peaks == 0: pad_turn_acc_peaks = 0
            else: pad_turn_acc_peaks = (max_turn_acc_peaks - mean_turn_acc_peaks)/std_turn_acc_peaks

            hard_turn_acc_count = np.sum((np.array(abs(ts_turn['acc_y']))[turn_acc_peaks])>1)
            abrupt_turn_acc_count = np.sum(turn_acc_width < 10)
            
        self.feat_set = np.concatenate([self.feat_set, np.array([max_turn_acc_peaks, pad_turn_acc_peaks, max_turn_acc, pad_turn_acc, hard_turn_acc_count, abrupt_turn_acc_count])]) 
        self.feat_names += ['max_turn_acc_peaks', 'pad_turn_acc_peaks', 'max_turn_acc', 'pad_turn_acc', 'hard_turn_acc_count', 'abrupt_turn_acc_count']

    def lane_change(self, ts_full):
        turn_window = 20
        turn_event =  np.array(self.ts.loc[(self.ts['Bearing_change']>30) ,'second'])
        turn_event_bound = []
        for i in turn_event:
            turn_event_bound+=(list(range(int(i-turn_window),int(i+turn_window))))
        turn_event_bound = list(set(turn_event_bound))

        ts_lane = ts_full.copy(deep = True)
        ts_lane.loc[ts_lane['second'].isin(turn_event_bound),'acc_y'] = 0

        lane_change_peaks, _ = find_peaks(abs(ts_lane['acc_y']) ,height = 0.33)
        if len(lane_change_peaks) == 0:
            mean_lane_change_acc = np.nan 
            max_lane_change_acc = np.nan 
            pad_lane_change_acc = np.nan 

            mean_lane_change_peaks = np.nan 
            max_lane_change_peaks = np.nan 
            pad_lane_change_peaks = np.nan 

            hard_lane_change_count = 0
            abrupt_lane_change_count = 0

        else:
            lane_change_peaks_val = np.array(abs(ts_lane['acc_y']))[lane_change_peaks]
            lane_change_width = peak_widths(abs(ts_lane['acc_y']) , lane_change_peaks)[0]
            lane_change_acc = lane_change_peaks_val/lane_change_width

            mean_lane_change_acc = np.mean(lane_change_acc) 
            max_lane_change_acc = np.max(lane_change_acc)
            std_lane_change_acc = np.std(lane_change_acc)

            if std_lane_change_acc == 0: pad_lane_change_acc = 0
            else: pad_lane_change_acc = (max_lane_change_acc - mean_lane_change_acc)/std_lane_change_acc

            mean_lane_change_peaks = np.mean(lane_change_peaks_val) 
            max_lane_change_peaks = np.max(lane_change_peaks_val)
            std_lane_change_peaks = np.std(lane_change_peaks_val)

            if std_lane_change_peaks == 0: pad_lane_change_peaks = 0
            else: pad_lane_change_peaks = (max_lane_change_peaks - mean_lane_change_peaks)/std_lane_change_peaks

            hard_lane_change_count = np.sum((np.array(abs(ts_lane['acc_y']))[lane_change_peaks])>1)
            abrupt_lane_change_count = np.sum(lane_change_width < 10)
            
        self.feat_set = np.concatenate([self.feat_set, np.array([max_lane_change_peaks, pad_lane_change_peaks, max_lane_change_acc, pad_lane_change_acc, hard_lane_change_count, abrupt_lane_change_count])])
        self.feat_names += ['max_lane_change_peaks', 'pad_lane_change_peaks', 'max_lane_change_acc', 'pad_lane_change_acc', 'hard_lane_change_count', 'abrupt_lane_change_count']
        
    def output_features(self):       
              
        trip_features = pd.DataFrame(self.feat_set)    
        trip_features = trip_features.T
        trip_features.columns = self.feat_names

        return trip_features
