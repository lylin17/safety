import numpy as np
from lime import lime_tabular
import random
import re

trip_idx = 0

class Feedback:
    '''
    Generate textual feedback from trained model using local interpretable model-agnostic explainations (LIME) 
    '''
    def __init__(self, model, X_train, X_val, y_val, features):
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.y_val = y_val
        self.features = features
        self.exp = None

    def feedback_text(self, trip_idx):
        random.seed(42)
        
        pred = self.model.predict(self.X_val[trip_idx:trip_idx+1,:])[0][1]
        print('Predicted Probability: {0}'.format(np.round(pred,3)))
        label = ['Safe', 'Dangerous'][self.y_val[trip_idx]]
        print('Label: {0}'.format(label))
        print()
        print('Trip Feedback:')

        explainer = lime_tabular.LimeTabularExplainer(self.X_train, feature_names = self.features, class_names = ['Safe','Dangerous'])
        self.exp = explainer.explain_instance(self.X_val[trip_idx,:], self.model.predict, num_features = 5)
        risks = [re.match('[0-9. <]*([A-Za-z_]*)[0-9. <]*',x[0]).group(1) for x in self.exp.as_list() if x[1] > 0]

        feedbacks = []
        for risk in risks:
            if 'speed' in risk: feedbacks.append('Slow down.')
            if 'forw' in risk: feedbacks.append('Accelerate gradually.')
            if 'brake' in risk: feedbacks.append('Smoother braking reccomended.')
            if 'turn' in risk: feedbacks.append('Slow down when turning.')
            if 'lane_change' in risk: feedbacks.append('Gradual lane change reccomended.')
            if 'veh_jerk' in risk: feedbacks.append('Reduce vehicular jerks during trip.')

        feedbacks = list(set(feedbacks))

        for feedback in feedbacks:
            print('   {0}'.format(feedback))
            
    def show_in_notebook(self):
        self.exp.show_in_notebook()
