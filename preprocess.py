import numpy as np
import pandas as pd

def load():
    raw_data = pd.read_csv('diabetes.csv')
    raw_data.head()

    raw_data.info()

    # Load the example Iris dataset for demonstration
    data = raw_data

    raw_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = raw_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

    # g = raw_data.hist(figsize=(20,20))

    raw_data.isnull().sum()

    # g = sns.pairplot(raw_data, hue="Outcome")

    def median_target(var):
        temp = raw_data[raw_data[var].notnull()]
        temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
        return temp

    median_target('Insulin')

    raw_data.loc[(raw_data['Outcome'] == 0 ) & (raw_data['Insulin'].isnull()), 'Insulin'] = 102.5
    raw_data.loc[(raw_data['Outcome'] == 1 ) & (raw_data['Insulin'].isnull()), 'Insulin'] = 169.5

    median_target('BMI')

    raw_data.loc[(raw_data['Outcome'] == 0 ) & (raw_data['BMI'].isnull()), 'BMI'] = 30.1
    raw_data.loc[(raw_data['Outcome'] == 1 ) & (raw_data['BMI'].isnull()), 'BMI'] = 34.3

    median_target('Glucose')

    raw_data.loc[(raw_data['Outcome'] == 0 ) & (raw_data['Glucose'].isnull()), 'Glucose'] = 107
    raw_data.loc[(raw_data['Outcome'] == 1 ) & (raw_data['Glucose'].isnull()), 'Glucose'] = 140

    median_target('BloodPressure')

    raw_data.loc[(raw_data['Outcome'] == 0 ) & (raw_data['BloodPressure'].isnull()), 'BloodPressure'] = 70.0
    raw_data.loc[(raw_data['Outcome'] == 1 ) & (raw_data['BloodPressure'].isnull()), 'BloodPressure'] = 74.5

    median_target('SkinThickness')

    raw_data.loc[(raw_data['Outcome'] == 0 ) & (raw_data['SkinThickness'].isnull()), 'SkinThickness'] = 27.0
    raw_data.loc[(raw_data['Outcome'] == 1 ) & (raw_data['SkinThickness'].isnull()), 'SkinThickness'] = 32.0

    raw_data.isnull().sum()

    # g = sns.pairplot(raw_data, hue="Outcome")
    return raw_data

def IQR(data):
    
    for feature in data:

        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)

        IQR = Q3-Q1
        lower = Q1- 1.5*IQR
        upper = Q3 + 1.5*IQR

        if data[(data[feature] > upper)].any(axis=None):
                print(feature,"yes")
        else:
                print(feature, "no")

        outliers = (data[feature] < lower) | (data[feature] > upper)
        
        data = data[~outliers]
        
    return data


def Z_score(data,threshold = 3):
    
    scores = (data - data.mean())/data.std()
    outlier_upper = np.where(scores > threshold)
    outlier_lower = np.where(scores < -1*threshold)
    
    data = data.drop(data.index[outlier_upper[0]])

    data = data.drop(data.index[outlier_lower[0]])

    return data


