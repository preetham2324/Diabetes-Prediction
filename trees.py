# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

from preprocess import load,IQR,Z_score
from sklearn.model_selection import GridSearchCV

# dataframe creation
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

#---------Base Values---------------
models = []
models.append(('Decision Tree Classifier', DecisionTreeClassifier(random_state = 42)))
models.append(('Random Forest Classifier', RandomForestClassifier(random_state = 42)))
models.append(('K Nearest Neighbour Classifier', KNeighborsClassifier()))
results = []
names = []

for name, model in models:
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        accuracy2 = accuracy_score(y_pred, y_test)
        msg = "%s: %f" % (name, accuracy2)
        print(msg)

###########################################################################################################################
########################################    Decission Tree Classifier    ##################################################    
###########################################################################################################################

#-------------------------------IQR------------------------------------------#

model_dtree1 = DecisionTreeClassifier()
params1 = {
    'splitter': ['best'],
    'max_depth': [None, 5,7, 10],
    'min_samples_split': [2,4, 5, 7,10,12,14],
    'min_samples_leaf': [2, 4,6,8],
    'max_leaf_nodes': [None, 5, 10, 20],
}
grid_search = GridSearchCV(model_dtree1,params1,cv=5)
grid_search.fit(X_IQR,y_IQR)
print(grid_search.best_params_)

#{'max_depth': None, 'max_leaf_nodes': None, 'min_samples_leaf': 2, 'min_samples_split': 12, 'splitter': 'best'}

best_model = DecisionTreeClassifier(**grid_search.best_params_).fit(X_IQR,y_IQR)
y_pred = best_model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy IQR for Decission Tree: {accuracy}")
print("Classification Report:\n", classification_report_str)

# #------------------------------Z_score---------------------------------------#

model_dtree2 = DecisionTreeClassifier()
params2 = {
    'splitter': ['best'],
    'max_depth': [None, 5,7, 10,12, 15],
    'min_samples_split': [2, 4, 5, 7, 10, 12, 14],
    'min_samples_leaf': [2, 4, 6, 8],
    'max_leaf_nodes': [None, 5, 10, 20],
}

grid_search = GridSearchCV(model_dtree2,param_grid=params2,cv=5)
grid_search.fit(X_Zscore,y_Zscore)
print(grid_search.best_params_)
#{'max_depth': None, 'max_leaf_nodes': None, 'min_samples_leaf': 8, 'min_samples_split': 2, 'splitter': 'best'}

best_model = DecisionTreeClassifier().fit(X_Zscore,y_Zscore)
y_pred = best_model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy Z_score for Decission Tree: {accuracy}sns.set()")
print("Classification Report:\n", classification_report_str)

###########################################################################################################################
########################################    Random Forest Classifier    ##################################################    
###########################################################################################################################

#-------------------------------IQR------------------------------------------#

# model_rf1 = RandomForestClassifier()
# params3 = {
#     'n_estimators': [150,165,175,185, 200],
#     'max_depth': [None, 5, 8, 10, 12, 15],
#     'min_samples_split': [2,4, 5,7,8],
#     'min_samples_leaf': [ 2, 3,4,6,8],
#     'max_features': ['auto', 'sqrt', 'log2'],
# }

# grid_search = GridSearchCV(model_rf1,params3,cv=5)
# grid_search.fit(X_IQR,y_IQR)
# print(grid_search.best_params_)
# best_model = RandomForestClassifier(**grid_search.best_params_).fit(X_IQR,y_IQR)
# # best_model = RandomForestClassifier().fit(X_IQR,y_IQR)
# y_pred = best_model.predict(X_test)
# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# classification_report_str = classification_report(y_test, y_pred)

# print(f"Accuracy IQR for RandomForest Classifier: {accuracy}")
# print("Classification Report:\n", classification_report_str)

# #------------------------------Z_score---------------------------------------#

model_rf2 = RandomForestClassifier()
params4 = {
    'n_estimators': [50, 100,150, 200,250, 300],
    'max_depth': [None, 5,8, 10,12, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4,6,8],
}

grid_search = GridSearchCV(model_rf2,params4,cv=5)
grid_search.fit(X_Zscore,y_Zscore)
print(grid_search.best_params_)
best_model = RandomForestClassifier(**grid_search.best_params_).fit(X_Zscore,y_Zscore)
# best_model = RandomForestClassifier().fit(X_IQR,y_IQR)
y_pred = best_model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy Z_score for RandomForest Classifier: {accuracy}")
print("Classification Report:\n", classification_report_str)

# ###########################################################################################################################
# ########################################    K Nearest Neighbour Classifier    ##################################################    
# ###########################################################################################################################

# #-------------------------------IQR------------------------------------------#

# model_knn1 = KNeighborsClassifier()
# params5 = {
#     'n_neighbors': [3, 4, 5, 6, 7, 8, 9],
#     'p': [1, 2],
# }

# grid_search = GridSearchCV(model_knn1,params5,cv=5)
# grid_search.fit(X_IQR,y_IQR)
# best_model = KNeighborsClassifier(**grid_search.best_params_).fit(X_IQR,y_IQR)
# y_pred = best_model.predict(X_test)
# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# classification_report_str = classification_report(y_test, y_pred)

# print(f"Accuracy IQR for KNN Classifier: {accuracy}")
# print("Classification Report:\n", classification_report_str)

# #------------------------------Z_score---------------------------------------#

# model_knn2 = KNeighborsClassifier()
# params6 = {
#     'n_neighbors': [3, 4, 5, 6, 7, 8, 9],
#     'weights': ['uniform', 'distance']
# }

# grid_search = GridSearchCV(model_knn2,params6,cv=5)
# grid_search.fit(X_Zscore,y_Zscore)
# best_model = KNeighborsClassifier(**grid_search.best_params_).fit(X_Zscore,y_Zscore)
# y_pred = best_model.predict(X_test)
# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# classification_report_str = classification_report(y_test, y_pred)

# print(f"Accuracy Z_score for KNN Classifier: {accuracy}")
# print("Classification Report:\n", classification_report_str)




