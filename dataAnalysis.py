import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl


excel_path = 'database_plot.xlsx'
column_names = ['tidal_volume/L','FVC/%','Central_airway_parameters/%','FEV1_FVC/%','PEF/%','PAP_A/%','PAP_B/%','PAP_C/%','PAP_D/%']
# formate data as we needed
data = pd.read_excel(excel_path)

print(data.describe())
print(data.head())
data[column_names].hist()
pl.show()