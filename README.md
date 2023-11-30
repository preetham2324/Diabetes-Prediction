# Diabetes-Prediction

## Description
In this project, we will be analysing a large number of machine learning models in order to be used in the field of Internet
of Medical Things (IoMT), specifically in the case of predicting diabetes mellitus (type 2 diabetes). Using the Pima
Indians Diabetes database in order to train and validate the models, we analyze the accuracy, precision, AUC and many
other metrics in order to evaluate the best working model for the task in hand. This project shows the significance
of machine learning in healthcare and predicive analysis, and can act as a secondary opinion to provide assistance to
workers of the medical field.

## Dataset
kaggle pima india dataset  
[diabetes.csv](diabetes.csv)

## Objective
The following points were the objective of the project 
  *  Descriptive Analysis
  *  Data Preprocessing
  *  Outlier Detection
  *  Model Evaluation
    to find the best model
    
## Files
[Booster.py](Booster.py) contains 2 ensemble models adaboost and xgboost  
[trees.py](trees.py) contains decision tree model and random forest model and Knn model  
[smote_lstm.py](smote_lstm.py) contains smote based lstm model  
[Lda_qda.py](Lda_Qda.py) containg the models lda and Qda  
[stacking.py](stacking.py) contains the models basic SVM,logistic models  
[requirements.txt](requirements.txt) contains the requirements



## Run
make sure to have all [requirements.txt](requirements.txt) installed  
run like any other python file  

```
     python3 file_name.py
```
gives the accuracy of models in files with basic preprocessing(handled missing values) and with IQR and Z_score methods and their confusion matrix
