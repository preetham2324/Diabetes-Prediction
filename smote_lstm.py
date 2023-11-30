import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from tensorflow import _keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import pandas as pd
from preprocess import load,IQR
from sklearn.metrics import accuracy_score,classification_report

def min_max_normalization(column):
    min_val = column.min()
    max_val = column.max()
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column

# Generating dummy imbalanced data
# X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, weights=[0.1, 0.9], random_state=42)

data = load()
attributes = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
for feature in data:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)

    IQR = Q3-Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    if (data[(data[feature] > upper)].any(axis=None)) or (data[(data[feature] < lower)].any(axis=None)):
        print(feature,"yes")
    else:
        print(feature, "no")
    
    data.loc[data[feature] > upper,feature] = upper
    data.loc[data[feature] < lower,feature] = lower



threshold =3
data1 = data.copy(deep=True)
scores = (data1 - data1.mean())/data1.std()

outlier_upper = np.where(scores > threshold)

outlier_lower = np.where(scores < -1*threshold)

mean = data1.mean()
std = data1.std()
#above gives indices as (x,y) in data matrix
#Removing rows x from upper 
# data1 = data1.drop(data1.index[outlier_upper[0]])
for x,y in zip(outlier_upper[0],outlier_upper[1]):
    data1.iloc[x, y] = 3*std[y] + mean[y]
# Removing rows with lower outliers
# data1 = data1.drop(data1.index[outlier_lower[0]])
for x,y in zip(outlier_lower[0],outlier_lower[1]):
     data1.iloc[x, y] = -3*std[y] + mean[y]

#---------------------------------------------------------------------

data2 = data.copy(deep=True)

X = data2.iloc[:, :-1].values
y = data2.iloc[:, -1].values

# Applying SMOTE to balance the data
smote = SMOTE(k_neighbors=5,random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Reshapeing the data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Building the LSTM model
model2 = Sequential()
model2.add(LSTM(units=96,input_shape=(X_train.shape[1], X_train.shape[2])))
model2.add(Dropout(0.5))
# model.add(LSTM(units=96,return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(units=96,return_sequences=True))
# model.add(Dropout(0.5))
model2.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
model2.compile(optimizer=Adam(learning_rate=0.005), loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model2.fit(X_train, y_train, epochs=100, batch_size=20)

# Evaluating the model
y_pred = model2.predict(X_test) >= 0.5 *1

accuracy2 = accuracy_score(y_test, y_pred)
classification_rep2 = classification_report(y_test, y_pred)

print(f'Accuracy smote normal : {accuracy2}')
print('Classification Report:\n', classification_rep2)


#---------------------------IQR-------------------#
# Apply min-max normalization to each column in the DataFrame
data = data.apply(min_max_normalization)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Applying SMOTE to balance the data
smote = SMOTE(k_neighbors=5,random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Reshapeing the data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=96,input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.5))
# model.add(LSTM(units=96,return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(units=96,return_sequences=True))
# model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.005), loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=100, batch_size=20)

# Evaluating the model
y_pred = model.predict(X_test) >= 0.5 *1

accuracy2 = accuracy_score(y_test, y_pred)
classification_rep2 = classification_report(y_test, y_pred)

print(f'Accuracy smote IQR : {accuracy2}')
print('Classification Report:\n', classification_rep2)


#---------------------------Z_score-------------------#

data1 = data1.apply(min_max_normalization)
X = data1.iloc[:, :-1].values
y = data1.iloc[:, -1].values

# Applying SMOTE to balance the data
smote = SMOTE(k_neighbors=5,random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Reshapeing the data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Building the LSTM model
model1 = Sequential()
model1.add(LSTM(units=96,input_shape=(X_train.shape[1], X_train.shape[2])))
model1.add(Dropout(0.5))
# model.add(LSTM(units=96,return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(units=96,return_sequences=True))
# model.add(Dropout(0.5))
model1.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
model1.compile(optimizer=Adam(learning_rate=0.005), loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model1.fit(X_train, y_train, epochs=100, batch_size=20)

# Evaluating the model
y_pred = model1.predict(X_test) >= 0.5 *1

accuracy2 = accuracy_score(y_test, y_pred)
classification_rep2 = classification_report(y_test, y_pred)

print(f'Accuracy smote Z_score : {accuracy2}')
print('Classification Report:\n', classification_rep2)