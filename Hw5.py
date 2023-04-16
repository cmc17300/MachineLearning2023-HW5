import pandas
from matplotlib import scale
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
# import re
from csv import reader

print("Question 1---------------: ")
# You can add the parameter data_home to wherever to where you want to download your data
dataset = pd.read_csv('CC GENERAL.csv')
features = ["CUST_ID", 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
            'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
            'CASH_ADVANCE_FREQUENCY'
    , 'CASH_ADVANCE_TRX']
# m = re.findall(features) #attempt to get this to work, unable to
x = dataset.loc[:, features]
y = dataset.loc[:, features]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Fit on training set only.
# scaler.fit(x)
x_fit = StandardScaler().fit(x)

# Apply transform to both the training set and the test set.
x_scaler = scaler.transform(x)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2, dataset[['Purchases']]], axis=1)
print(finaldf)
