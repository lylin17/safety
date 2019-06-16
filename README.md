# Title

overview

## Solution

#### Prerequisites:

1. Before running notebooks and scripts in this repository, install dependencies using pip:

<pre><code>pip install -r requirements.txt</code></pre>

2. Download preprocess.csv, model.h5 and model_full.h5 from XXXX and put it in this repository folder
3. Place the 11 raw csv files provided for the challenge (10 feature csv and 1 label csv) in this repository folder
4. Place the hold-out feature (1 file, named test_features.csv) and label (1 file, named test_labels.csv) files in this repository folder

#### Solution Details and Scripts

1. To reproduced dataset with engineered features (preprocess.csv, downloaded from XXXXX as specified in Prerequisites point 2), run preprocess_train.py script as follows:

<pre><code>python preprocess_train.py</code></pre>

2. Detailed description of the solution provided in solution.ipynb

3. To reproduce the final model file trained on the full data (model_full.h5, downloaded from XXXXX as specified in Prerequisites point 2), run train.py script as follows:

<pre><code>python train.py</code></pre>

#### Evaluation on hold-out set

To perform preprocessing, feature engineering and model inference on the hold-out set (setup as described in Prerequisites, point 4), run test.py script as follows:

<pre><code>python test.py</code></pre>

- prediction.csv: predicted probability for each bookingID
- evaluation.txt: recall, precision, f1 score, roc-auc and average precision for hold-out set

## Built With

Code tested with python 3.5.5 running on Azure Data Science Virtual Machine (Ubuntu 16.04)

## Author

<p>Lin Laiyi, Senior AI Apprentice at AI Singapore, NUS MSBA 2007/2008</p>
<p>LinkedIn: https://www.linkedin.com/in/laiyilin/</p>
<p>Portfolio of selected analytics project: https://drive.google.com/file/d/1fVntFEvj6us_6ERzRmbU85EOeZymFxEm/view</p>