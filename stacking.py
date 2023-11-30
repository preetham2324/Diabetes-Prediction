
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression,SGDOneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from preprocess import load,IQR,Z_score
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


raw_data = load()
train, test= train_test_split(raw_data, test_size=0.2, random_state=42)
train_IQR = IQR(train)
train_Z = Z_score(train)

X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

X_IQR = train_IQR.iloc[:, :-1].values
y_IQR = train_IQR.iloc[:, -1].values
X_Zscore = train_Z.iloc[:,:-1].values
y_Zscore = train_Z.iloc[:, -1].values

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

#################################################################################

# Logistic Regression Base Model

model=LogisticRegression()
grid = {"C": np.logspace(-2, 2, 5), "penalty": ["l2"], "max_iter": [500, 1000, 1500]}
# l1 lasso l2 ridge
model=GridSearchCV(model,grid,cv=2)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
cm=confusion_matrix(y_test,y_pred)

print(f"Accuracy of Logistic Base Model: {accuracy}")
print("Clonfusion Matrix:\n", cm)
print("Classification Report:\n", classification_report_str)

#################################################################################

# Logistic Regression IQR Model

model=LogisticRegression()
grid = {"C": np.logspace(-2, 2, 5), "penalty": ["l2"], "max_iter": [500, 1000, 1500]}
# l1 lasso l2 ridge
model=GridSearchCV(model,grid,cv=2)
model.fit(X_IQR,y_IQR)
y_pred=model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
cm=confusion_matrix(y_test,y_pred)
print(f"Accuracy of Logistic IQR Model: {accuracy}")
print("Clonfusion Matrix:\n", cm)
print("Classification Report:\n", classification_report_str)

#################################################################################

# Logistic Regression Z-Score Model

model=LogisticRegression()
grid = {"C": np.logspace(-2, 2, 5), "penalty": ["l2"], "max_iter": [500, 1000, 1500]}
# l1 lasso l2 ridge
model=GridSearchCV(model,grid,cv=2)
model.fit(X_Zscore,y_Zscore)
y_pred=model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
cm=confusion_matrix(y_test,y_pred)
print(f"Accuracy of Logistic Z-Score Model: {accuracy}")
print("Clonfusion Matrix:\n", cm)
print("Classification Report:\n", classification_report_str)
#################################################################################

# SVM Base Model

model=SVC()
grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'scale', 'auto'],
    
}
model=GridSearchCV(model,grid,cv=2)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
cm=confusion_matrix(y_test,y_pred)
print(f"Accuracy of SVM Base Model: {accuracy}")
print("Clonfusion Matrix:\n", cm)
print("Classification Report:\n", classification_report_str)

#################################################################################

# SVM IQR Model

model=SVC()
grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'scale', 'auto'],
    
}
model=GridSearchCV(model,grid,cv=2)
model.fit(X_IQR,y_IQR)
y_pred=model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
cm=confusion_matrix(y_test,y_pred)
print(f"Accuracy of SVM IQR Model: {accuracy}")
print("Clonfusion Matrix:\n", cm)
print("Classification Report:\n", classification_report_str)

#################################################################################

# SVM Z-Score Model

model=SVC()
grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'scale', 'auto'],
    
}
model=GridSearchCV(model,grid,cv=2)
model.fit(X_Zscore,y_Zscore)
y_pred=model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
cm=confusion_matrix(y_test,y_pred)
print(f"Accuracy of SVM Z_Score Model: {accuracy}")
print("Clonfusion Matrix:\n", cm)
print("Classification Report:\n", classification_report_str)

#################################################################################


# train, val_train, test, val_test = train_test_split(X, y, test_size = 0.5, random_state = 42)
# X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state = 42)
# print(X_train)

# #################################################################################

# # Stacking Base  Model

# model=LogisticRegression()
# grid = {"C": np.logspace(-2, 2, 5), "penalty": ["l2"], "max_iter": [500, 1000, 1500]}
# # l1 lasso l2 ridge
# lr=GridSearchCV(model,grid,cv=2)
# lr.fit(X_train, y_train)

# svm = SVC()
# svm.fit(X_train, y_train)

# predict_val1 = lr.predict(val_train)
# predict_val2 = svm.predict(val_train)

# predict_val = np.column_stack((predict_val1, predict_val2))
# predict_test1 = lr.predict(X_test)
# predict_test2 = svm.predict(X_test)
# predict_test = np.column_stack((predict_test1, predict_test2))

# rand_clf = RandomForestClassifier()
# rand_clf.fit(predict_val, val_test)
# stacking_acc = accuracy_score(y_test, rand_clf.predict(predict_test))
# print(f"Accuracy of Stacking Model: {stacking_acc}")
# classification_report_str = classification_report(y_test,rand_clf.predict(predict_test) )
# cm=confusion_matrix(y_test,rand_clf.predict(predict_test))
# print("Confusion Matrix:\n", cm)
# print("Classification Report:\n", classification_report_str)


#################################################################################




