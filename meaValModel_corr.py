import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd


excel_path = 'database_en.xlsx'
# column_names = ['const','FVC','Central_airway_parameters','FEV1/FVC','PEF','PAP_A','PAP_B']
column_names = ['const','FVC','tidal_volume','Central_airway_parameters','FEV1_FVC','PEF','PAP_A','PAP_B','PAP_C','PAP_D']
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
# fill the NaN with median 1.31
data['tidal_volume'].fillna(data['tidal_volume'].median(),inplace=True)
data.to_excel('test.xlsx', index=False)

result_dic = {}
result_dic[''] = {'Correlation':'Correlation','Pvalue':'Pvalue'}
for i in column_names: 
    result = scipy.stats.pearsonr(data[i], data['disease'])
    result_dic[i] = {'Correlation':result[0],'Pvalue':result[1]}
    print(result_dic)

df = pd.DataFrame(result_dic)
df.to_excel('MeaVal_corrPvalue.xlsx', index=False)

import seaborn as sns
corr = data[column_names].corr(method='spearman')
cmap = sns.diverging_palette(250, 10, n=3, sep=10,as_cmap=True)
sns.set(font_scale=0.5)

plt.subplots(figsize=(6, 6))  # 设置画面大小
sns.heatmap(corr, annot=True, vmax=1, square=True, linewidths=0.3 ,cmap=cmap)
plt.show() 