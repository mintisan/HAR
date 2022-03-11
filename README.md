
# DL baseline for Human Activity Recognition

## Dependencies
- torch
- keras
- tensorflow
- netron
- torchsummary
- seaborn
- sklearn
- graphviz : pydot/pydotplus/graphviz


## Step
1. Install the packages above.
2. Download [UCI HAR Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip) and extract to current directory
```
.gitignore
keras_cnn_har.py
pytorch_cnn_har.py
pytorch_model1.py
README.md
UCI HAR Dataset/
uci_data_loader.py
```
3. Get the results
```
python keras_cnn_har.py
python pytorch_cnn_har.py
python keras_lstm2d_har.py
python keras_lstm_har.py
python keras_gru_har.py
```