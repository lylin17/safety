import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks , peak_widths

class Reorientate_axis:
    '''
    Reorientate acclerometer data from smartphone frame of reference to vehicle frame of reference
    '''
    
    def __init__(self, ts, low_pass_cutoff):
        
        #Low pass filter for smoothing acclerometer data
        fs = 1
        nyquist = fs / 2
        cutoff = low_pass_cutoff * nyquist

        b, a = butter(5, cutoff, btype='lowpass')

        ts['acceleration_x'] = filtfilt(b, a, ts['acceleration_x'])
        ts['acceleration_y'] = filtfilt(b, a, ts['acceleration_y'])
        ts['acceleration_z'] = filtfilt(b, a, ts['acceleration_z'])

        #Normalize acceleration vectors by g = 9.81
        ts['acceleration_x'] = ts['acceleration_x']/9.81
        ts['acceleration_y'] = ts['acceleration_y']/9.81
        ts['acceleration_z'] = ts['acceleration_z']/9.81

        #Placeholder columns for reoriented acceleration vectors
        ts['acc_x'] = 0
        ts['acc_y'] = 0
        ts['acc_z'] = 0   
        
        self.ts = ts
        self.ts_blk = None

    def rotation(self):   

        #median value of window method instead of stationary/constant speed vehicle method
        a_x = self.ts_blk['acceleration_x'].median(axis=0)
        a_y = self.ts_blk['acceleration_y'].median(axis=0)
        a_z = self.ts_blk['acceleration_z'].median(axis=0)

        #round off a_z > 1g or a_z < -1g to be a_z = 1g or a_z = -1g for valid value in theta_tilt 
        if a_z > 1: a_z  = 1
        elif a_z < -1: a_z = -1  

        theta_tilt = np.arccos(a_z)
        phi_pre = np.arctan2(a_y,a_x)

        #sharp forward acceleration/decceleration on period selected with no large lateral accelerations

        ts1 = self.ts_blk.loc[(abs(self.ts_blk['acceleration_x'])<0.2),:].reset_index(drop=True)

        # Relaxing of constraint on lateral acceleration if period of no large lateral acceleration not found
        if ts1.shape[0] == 0 : ts1 = self.ts_blk.loc[(abs(self.ts_blk['acceleration_x'])<0.5),:].reset_index(drop=True)
        if ts1.shape[0] == 0 : ts1 = self.ts_blk

        max_acc = np.max(abs(ts1['forw_acc']))
        decel_second = ts1.loc[abs(ts1['forw_acc']) == max_acc,'second'].tolist()[0]

        #Allow 3s window for lag time in GPS estimates
        period_sel = ts1.loc[(ts1['second'] >= decel_second) & (ts1['second'] <= decel_second + 3),:]

        decel_a_x = period_sel['acceleration_x'].mean(axis=0)
        decel_a_y = period_sel['acceleration_y'].mean(axis=0)
        decel_a_z = period_sel['acceleration_z'].mean(axis=0)

        phi_post = np.arctan2((-decel_a_x*np.sin(phi_pre)+decel_a_y*np.cos(phi_pre)),(((decel_a_x*np.cos(phi_pre)+decel_a_y*np.sin(phi_pre))*np.cos(theta_tilt)) - decel_a_z*np.sin(theta_tilt)))

        #Rotation matrix from euler angles
        R_Z_post = np.array([np.cos(phi_post),np.sin(phi_post),0,-np.sin(phi_post),np.cos(phi_post),0,0,0,1]).reshape(3,3)
        R_Y = np.array([np.cos(theta_tilt),0,-np.sin(theta_tilt),0,1,0,np.sin(theta_tilt),0,np.cos(theta_tilt)]).reshape(3,3)
        R_Z_pre = np.array([np.cos(phi_pre),np.sin(phi_pre),0,-np.sin(phi_pre),np.cos(phi_pre),0,0,0,1]).reshape(3,3)

        #Axis rotations by matrix multiplication with rotation matrices
        pre =  np.matmul(np.broadcast_to(R_Z_pre,(self.ts_blk.shape[0],3,3)),pd.concat([self.ts_blk['acceleration_x'],self.ts_blk['acceleration_y'],self.ts_blk['acceleration_z']],axis=1).values.reshape(self.ts_blk.shape[0],3,1))
        tilt = np.matmul(np.broadcast_to(R_Y,(self.ts_blk.shape[0],3,3)),pre)
        new_accel = np.matmul(np.broadcast_to(R_Z_post,(self.ts_blk.shape[0],3,3)),tilt).reshape(self.ts_blk.shape[0],3)
        
        return new_accel

    def reorientate(self, window):

        # Get time jump in timeseries        
        timejump = []
        for i in range (self.ts.shape[0]-1):
            if (self.ts.loc[i+1,'second'] - self.ts.loc[i,'second']) > 5 :
                if (i+1)%window != 0: timejump.append(i+1)

        #Divide time series into blocks of specific window size
        #Get Euler angles and perform axis reorientation block-by-block
        idx_list = list(range(0,self.ts.shape[0],window))
        idx_list.append(self.ts.shape[0])
        idx_list += timejump
        idx_list.sort()

        for idx in range(len(idx_list)-1):

            self.ts_blk = self.ts.iloc[idx_list[idx]:idx_list[idx+1],:].reset_index(drop=True)    
            new_accel = self.rotation()

            #Undo previous normlization by g = 9.81
            new_accel = new_accel * 9.81

            self.ts.iloc[idx_list[idx]:idx_list[idx+1],self.ts.columns.tolist().index('acc_x')] = new_accel[:,0]
            self.ts.iloc[idx_list[idx]:idx_list[idx+1],self.ts.columns.tolist().index('acc_y')] = new_accel[:,1]
            self.ts.iloc[idx_list[idx]:idx_list[idx+1],self.ts.columns.tolist().index('acc_z')]  = new_accel[:,2]