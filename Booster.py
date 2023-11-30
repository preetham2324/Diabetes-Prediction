import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from preprocess import load,IQR,Z_score
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.model_selection import KFold
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten,Dropout
from keras.optimizers import Adam
import pandas as pd

raw_data = load()
train, test= train_test_split(raw_data, test_size=0.2, random_state=42)
train_IRQ = IQR(train)
train_Z = Z_score(train)

X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values
X_IRQ = train_IRQ.iloc[:, :-1].values
y_IRQ = train_IRQ.iloc[:, -1].values
X_Zscore = train_Z.iloc[:,:-1].values
y_Zscore = train_Z.iloc[:, -1].values

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

#---------Base Values---------------
models = []
models.append(('XGB', GradientBoostingClassifier(random_state = 42)))
models.append(("Adaboost", AdaBoostClassifier(random_state = 42)))
results = []
names = []

for name, model in models:
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        accuracy2 = accuracy_score(y_pred, y_test)
        msg = "%s: %f" % (name, accuracy2)
        print(msg)
       
#---------------Basic CNN------------------------

X_train = X_IRQ.reshape((X_IRQ.shape[0], X_IRQ.shape[1], 1))
X_test1 = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu',padding='same',input_shape=(X_train.shape[1], 1)))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu',padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary
model.fit(X_train, y_IRQ, epochs=200, batch_size=20)

# Evaluate the model on test data
predictions = model.predict(X_test1)
predictions = predictions.round()

accuracy2 = accuracy_score(y_test, predictions)
classification_rep2 = classification_report(y_test, predictions)

print(f'Accuracy CNN : {accuracy2}')
print('Classification Report:\n', classification_rep2)


#----------------IQR----------------------
xgb = GradientBoostingClassifier()
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'learning_rate': [0.1, 0.2, 0.3, 0.4],
    #'subsample': [1,0.75,0.5,0.25]
}

xgb_cv_model  = GridSearchCV(xgb,param_grid= param_grid, cv=5,scoring='accuracy',verbose=0).fit(X_IRQ, y_IRQ)
# best_params  = xgb_cv_model.best_params_
xgb_tuned =  GradientBoostingClassifier(**xgb_cv_model.best_params_).fit(X_IRQ,y_IRQ)

#print(xgb_cv_model.best_params_)
#{'learning_rate': 0.4, 'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 100}

y_pred = xgb_tuned.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy IQR xgb: {accuracy}")
print("Classification Report:\n", classification_report_str)

#-------------Z_score---------------
xgb1 = GradientBoostingClassifier()
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'learning_rate': [0.1, 0.2, 0.3, 0.4],
    
}

xgb_cv_model  = GridSearchCV(xgb1,param_grid= param_grid, cv=5,scoring='accuracy').fit(X_Zscore, y_Zscore)
# best_params  = xgb_cv_model.best_params_
#{'learning_rate': 0.4, 'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 200}
xgb1 =  GradientBoostingClassifier(**xgb_cv_model.best_params_).fit(X_Zscore,y_Zscore)

print(xgb_cv_model.best_params_)

y_pred = xgb1.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy Z_score xgb: {accuracy}")
print("Classification Report:\n", classification_report_str)


#--------------Adaboost with IQR-----------------------

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1.0],
    # 'algorithm': ['SAMME', 'SAMME.R']
}
adaboost_model1 = AdaBoostClassifier( random_state=42)
grid_search = GridSearchCV(adaboost_model1, param_grid, cv=5, scoring='accuracy').fit(X_IRQ, y_IRQ)
# adaboost_model2.fit(X_Zscore, y_Zscore)
adaboost_model1 =  AdaBoostClassifier(**grid_search.best_params_).fit(X_IRQ,y_IRQ)

# print(grid_search.best_params_)
#{'learning_rate': 0.1, 'n_estimators': 100}

y_pred1 = adaboost_model1.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred1)
classification_rep1 = classification_report(y_test, y_pred1)

print(f'Accuracy IQR Adaboost : {accuracy1}')
print('Classification Report:\n', classification_rep1)


#----------------Adaboost with Z_score------------------

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1.0],
    # 'algorithm': ['SAMME', 'SAMME.R']
}
adaboost_model2 = AdaBoostClassifier( random_state=42)
grid_search = GridSearchCV(adaboost_model2, param_grid, cv=5, scoring='accuracy').fit(X_Zscore, y_Zscore)
# adaboost_model2.fit(X_Zscore, y_Zscore)
adaboost_model2 =  AdaBoostClassifier(**grid_search.best_params_).fit(X_Zscore,y_Zscore)

# print(grid_search.best_params_)
# {'learning_rate': 0.1, 'n_estimators': 150}
y_pred2 = adaboost_model2.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)
classification_rep2 = classification_report(y_test, y_pred2)

print(f'Accuracy Z_score Adaboost : {accuracy2}')
print('Classification Report:\n', classification_rep2)

