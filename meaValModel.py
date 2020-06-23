# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import pylab as pl
import pandas as pd


excel_path = 'database_en.xlsx'
# column_names = ['const','FVC','Central_airway_parameters','FEV1_FVC','PEF','PAP_A','PAP_B']
column_names = ['const','disease','tidal_volume','FVC','Central_airway_parameters','FEV1_FVC','PEF','PAP_A','PAP_B','PAP_C','PAP_D']
column_names = ['const','tidal_volume','Central_airway_parameters','FEV1_FVC','PEF','PAP_A','PAP_B','PAP_C','PAP_D']
column_names = ['const','FEV1_FVC','PAP_A','PAP_B']
column_names = ['const','PEF','age','FEV1_FVC','PAP_A','PAP_B']


# formate data as we needed
dataset_neg = pd.read_excel(excel_path, sheet_name='阴性')
del dataset_neg['编号']
dataset_neg['disease']= 0
dataset_pos = pd.read_excel(excel_path, sheet_name='阳性')
del dataset_pos['编号']
dataset_pos['disease']= 1

# add y tag 
data = pd.concat([dataset_neg,dataset_pos],axis=0,ignore_index=True)
data = pd.concat([data, pd.get_dummies(data['sexy'])], axis=1)
data.rename(columns={'男':'male'},inplace=True)
del data['sexy']
del data['女']
# add constant
data = sm.add_constant(data)
# add BMI
data['BMI'] = data['weight']/(data['height']*data['height'])*10000
data['tidal_volume'].fillna(data['tidal_volume'].median(),inplace=True)
data.to_excel('newDataFrame.xlsx', index=False)

# split data to train and test
X = data[column_names]
y = data['disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# training model
logreg = sm.Logit(y_train,X_train)
logreg_res = logreg.fit()
print(logreg_res.summary())


# prediction
y_pred = logreg_res.predict(X_test)
y_pred_int =(y_pred>0.5).astype('int')
compare=pd.DataFrame({'predictedValue':y_pred,'predictedValueInt':y_pred_int,'actualValue':y_test})
compare.to_excel('compare.xlsx', index=False)

# prediction accurancy 
acc = sum(y_pred_int == y_test) /np.float(len(y_test))
print('The accurancy is %.2f' %acc)

# calculate variable correlation and choose the best group
def vif(df, col_i):
    from statsmodels.formula.api import ols

    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)

for i in X.columns:
    print(i, '\t', vif(df=X, col_i=i))

