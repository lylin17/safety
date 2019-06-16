# Title

####Overview
blah blah blah

####Setup Required:

1. Before running codes in this repository, install dependencies using pip:

<pre><code>pip install -r requirements.txt</code></pre>

2. Download preprocess.csv, model.h5 and model_full.h5 from XXXX and put it in this repository folder
3. Place the 11 raw csv files (10 feature csv and 1 label csv) in this repository folder
4. Place the hold-out feature (1 file, named test_features.csv) and label (1 file, named test_labels.csv) in this repository

### Solution

Detailed description of the solution provided in solution.ipynb

To get the final model file trained on the full data provided for the challenge (model_full.h5 alternatively download from  as specified in Setup Required, point 2), run train.py script as follows:

<pre><code>python train.py</code></pre>

### Evaluation on hold-out set

To perform preprocessing, feature engineering and model inference , run test.py script as follows:

<pre><code>python test.py</code></pre>

prediction.csv: predicted probabilities for each bookingID
evaluation.txt: recall, precision and f1 score at classification threshold = 0.5; roc-auc and average precision 