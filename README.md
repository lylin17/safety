# Title

#### Overview
blah blah blah

#### Setup Required:

1. Before running codes in this repository, install dependencies using pip:

<pre><code>pip install -r requirements.txt</code></pre>

2. Download preprocess.csv, model.h5 and model_full.h5 from XXXX and put it in this repository folder
3. Place the 11 raw csv files provided for the challenge (10 feature csv and 1 label csv) in this repository folder
4. Place the hold-out feature (1 file, named test_features.csv) and label (1 file, named test_labels.csv) files in this repository

#### Solution

To reproduced data set with engineered features (preprocess.csv, downloaded from XXXXX as specified in Setup Required, point 2), run preprocess_train.py script as follows:

<pre><code>python train.py</code></pre>

**Detailed description of the solution provided in solution.ipynb**

To reproduce the final model file trained on the full data (model_full.h5, downloaded from XXXXX as specified in Setup Required, point 2), run train.py script as follows:

<pre><code>python train.py</code></pre>

#### Evaluation on hold-out set

To perform preprocessing, feature engineering and model inference on the hold-out set, run test.py script as follows:

<pre><code>python test.py</code></pre>

<p>prediction.csv: predicted probabilities for each bookingID</p>
<p>evaluation.txt: recall, precision and f1 score at classification threshold = 0.5; roc-auc and average precision</p> 