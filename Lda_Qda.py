import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score,classification_report
from preprocess import load,IQR,Z_score

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

raw_data = load()
train, test= train_test_split(raw_data, test_size=0.2, random_state=42)
train_IQR = IQR(train)
train_Z = Z_score(train)

X = train.iloc[:, :-1].values
y = train.iloc[:, -1].values

X_IQR = train_IQR.iloc[:, :-1].values
y_IQR = train_IQR.iloc[:, -1].values
X_Zscore = train_Z.iloc[:,:-1].values
y_Zscore = train_Z.iloc[:, -1].values

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values


# Instantiate the GridSearchCV object
param_grid = {
    'solver': ['svd', 'lsqr', 'eigen'],  # Different solvers for LDA
    # You can explore other available parameters here, but 'solver' is one of the few tunable ones.
    # 'solver': ['svd', 'lsqr', 'eigen'],
    # 'shrinkage': [None, 'auto'],
    # 'priors': [None, [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]],  # Adjust priors if needed
    # 'reg_param': [0.0, 0.1, 0.5, 1.0],  # Regularization parameter
    'store_covariance': [True, False]
}


#################################################################################

# LDA BASE MODEL

lda_base = LinearDiscriminantAnalysis()
grid_search = GridSearchCV(lda_base, param_grid, cv=5)

grid_search.fit(X, y)

best_lda = grid_search.best_estimator_
y_pred = best_lda.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy of LDA Base Model: {accuracy}")
print("Classification Report:\n", classification_report_str)

#################################################################################

# LDA IQR MODEL

# Create an LDA model
lda1 = LinearDiscriminantAnalysis()
grid_search = GridSearchCV(lda1, param_grid, cv=5)


# Fit the model on the training data
grid_search.fit(X_IQR, y_IQR)

# Make predictions on the test data
best_lda = grid_search.best_estimator_
y_pred = best_lda.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy of LDA IQR: {accuracy}")
print("Classification Report:\n", classification_report_str)



#################################################################################

# LDA Z-SCORE MODEL


# Create an LDA model
lda2 = LinearDiscriminantAnalysis()
grid_search = GridSearchCV(lda2, param_grid, cv=5)

# Fit the model on the training data
grid_search.fit(X_Zscore, y_Zscore)

# Make predictions on the test data
best_lda = grid_search.best_estimator_
y_pred = best_lda.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy of LDA Z-score: {accuracy}")
print("Classification Report:\n", classification_report_str)


#################################################################################

param_grid = {
    'reg_param': [0.0, 0.1, 0.5, 1.0],  # Regularization parameter
    # 'priors': [None, [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]],  # Class priors
    'store_covariance': [True, False]
}

# QDA BASE MODEL

qda_base = QuadraticDiscriminantAnalysis()
grid_search = GridSearchCV(qda_base, param_grid, cv=5)


grid_search.fit(X, y)


best_qda = grid_search.best_estimator_
y_pred = best_qda.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy of QDA Base Model: {accuracy}")
print("Classification Report:\n", classification_report_str)



#################################################################################


# Create an QDA model
qda1 = QuadraticDiscriminantAnalysis()
grid_search = GridSearchCV(qda1, param_grid, cv=5)

# Fit the model on the training data
grid_search.fit(X_IQR, y_IQR)

# Make predictions on the test data
best_qda = grid_search.best_estimator_
y_pred = best_qda.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy of QDA IQR: {accuracy}")
print("Classification Report:\n", classification_report_str)


#################################################################################


# Create an QDA model
qda2 = QuadraticDiscriminantAnalysis()
grid_search = GridSearchCV(qda2, param_grid, cv=5)

# Fit the model on the training data
grid_search.fit(X_Zscore, y_Zscore)

# Make predictions on the test data
best_qda = grid_search.best_estimator_
y_pred = best_qda.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f"Accuracy of QDA Z-score: {accuracy}")
print("Classification Report:\n", classification_report_str)


####################################################
