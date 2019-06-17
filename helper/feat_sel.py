import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import scipy.stats as st

class Feat_sel:
    '''
    Perform feature selecion using Boruta and Lasso to isolate relevant feature set 
    '''
    
    def __init__(self, preprocess, X_train, y_train):

        self.features = preprocess.columns[1:]
        self.X_train = X_train
        self.y_train = y_train
        
    def boruta(self):
        rf = RandomForestClassifier(n_estimators = 500, random_state= 42)
        feat_selector = BorutaPy(rf, n_estimators='auto', perc = 70, random_state=42)
        feat_selector.fit(self.X_train, self.y_train)

        feature_sel = self.features[feat_selector.support_]
        
        return feature_sel
    
    def lasso(self):
        N = 500

        coef = pd.DataFrame(np.zeros((N,self.X_train.shape[1])),columns = self.features)
        lasso = SGDClassifier(loss='log', penalty='l1',max_iter = 5000, tol = 1e-6, random_state = 42) 

        for n in range(N):
            idx = np.random.choice(list(range(self.X_train.shape[0])),size = self.X_train.shape[0], replace = True)
            X_bootstrap = self.X_train[idx,:]
            y_bootstrap = self.y_train[idx]

            lasso.fit(X_bootstrap,y_bootstrap)
            coef.iloc[n,:] = lasso.coef_

        #Select features where 0 is not within the 95% confidence interval of the lasso coefficient distribution
        lower, upper = st.t.interval(0.95, N-1, loc=coef.mean(), scale=coef.std())
        lower[np.isnan(lower)] = 0
        upper[np.isnan(upper)] = 0

        feature_sel = self.features[~((lower <= 0) * (upper >= 0))]
        
        return feature_sel

    def make_feat_set(self, feature_sel1, feature_sel2, X_val):

        feature_sel = np.union1d(feature_sel1, feature_sel2)

        feature_sel_idx = [self.features.tolist().index(feat) for feat in feature_sel]
        feature_sel_idx.sort()

        features_selected = np.array(self.features)[feature_sel_idx]
        np.save('feat_sel.npy',features_selected)

        X_train = self.X_train[:,feature_sel_idx]
        X_val = X_val[:,feature_sel_idx]
        
        return X_train, X_val, features_selected