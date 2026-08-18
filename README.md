# heart-rate-predictor
Use Machine Learning models to predict heart rate and blood glucose by using data from non invasive wearable devices.
The Train and Test dataset for predicting heart rate is [this](https://www.kaggle.com/datasets/amangopalgandhi/physiocgm-data-subset), which is a small subset of PhysioCGM dataset.

The way to run the linear regression code file is 
```
python3 linear_reg.py train.csv test1.csv predictions.txt weights.txt

```
where train and test csv files contain the data, the test data file should not have the heart rate column in the end. But in the test.csv in kaggle, it contains the heart rate column so as to later on check how the model performance was. So one should make a new test1.csv file by removing the last column from the test.csv file provided in  kaggle dataset.
