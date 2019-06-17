# Prompt Safety Feedback for Grab drivers 

- **Overview**: To generate prompt feedback on the driving behavior of Grab drivers at the end of each trip    
- **Business outcome**: Promote safe driving culture among Grab drivers and reduce customer complaints regarding dangerous driving. Act as a deterrent against dangerous manoeuvres if the drivers know that their driving behaviors are being monitored.    
- **Value-add**: Currently, Grab provide drivers with weekly feedback of their driving patterns (speed, braking, acceleration etc). This solution could potentially add new features to this feedback report. In addition, prompt feedback (sms or app pop-up) after each trip is proposed. When the trip is still fresh in the driver's mind, such prompt feedback could be more effective than an accumulated report at the end of the week.   
- **Implementation**:
    - *Inputs*: Telematics data collected from smart phones sensors during the trip
    - *Outputs*: Concrete textual feedback at the end of the trip on how to make the trip safer using local interpretable model-agnostic explainations (LIME)
    - *Feature Engineering Choices*: Careful choices were made during feature engineering to ensure that the model is not biased towards features that do not make sense. Such choices were made in the interest of generalizability even if it resulted in some loss in model performance observed for this particular dataset. 
    - *Model Choice*: Model interpretability was chosen over performance by employing a feature engineering-based approach (capture driving behaviors as features) compared to a black box RNN-type time series classification model  

## Solution

#### Prerequisites:

1. Before running notebooks and scripts in this repository, install dependencies using pip:

<pre><code>pip install -r requirements.txt</code></pre>

2. Download preprocess.csv, model.h5, model_full.h5 from https://drive.google.com/drive/folders/1PADzFR8T5pVn0S9kmqB6598e9SHS2EZl?usp=sharing and put it in this repository folder
3. Place the 11 raw csv files provided for the challenge (10 feature csv and 1 label csv, keeping their original filenames) in this repository folder
4. Place the hold-out feature file (1 csv file, named test_features.csv) and label file (1 csv file, named test_labels.csv) in this repository folder. Hold-out csv files should have identical columns as the ones provided for the challenge.

#### Solution Details and Scripts

1. Detailed description of the solution provided in solution.ipynb
	- Notebook outputs data_min.npy, data_range.npy and feat_sel.npy which are used by train.py and test.py
	- Notebook reproduces model.h5 (Prerequisites point 2) 

2. To reproduced dataset with engineered features (preprocess.csv, Prerequisites point 2), run preprocess_train.py script as follows:

<pre><code>python preprocess_train.py</code></pre>

3. To reproduce the final model file trained on the full data (model_full.h5, Prerequisites point 2), run train.py script as follows:

<pre><code>python train.py</code></pre>

#### Evaluation on hold-out set

To perform preprocessing, feature engineering and model inference on the hold-out set (setup as described in Prerequisites point 4), run test.py script as follows:

<pre><code>python test.py</code></pre>

- prediction.csv: predicted probability for each bookingID in the hold-out set
- evaluation.txt: recall, precision, f1 score, roc-auc and average precision for hold-out set

## Built With

Code tested with python 3.5.5 running on Azure Data Science Virtual Machine (Ubuntu 16.04)

## Author

<p>Lin Laiyi, Senior AI Apprentice at AI Singapore, NUS MSBA 2007/2008</p>
<p>LinkedIn: https://www.linkedin.com/in/laiyilin/</p>
<p>Portfolio of selected analytics project: https://drive.google.com/file/d/1fVntFEvj6us_6ERzRmbU85EOeZymFxEm/view</p>