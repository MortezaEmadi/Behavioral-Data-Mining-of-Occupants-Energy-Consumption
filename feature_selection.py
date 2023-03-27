"""
All Rights Reserved
@author Morteza Emadi
this was my old coding in Jupyter for implementing Mutual Information algorithm on
the processed features of 39 Homes
"""

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.feature_selection import mutual_info_regression


X=[]
Y=[]
om = pd.read_csv(r'data\normalized_and_regressed_1401.csv')
pd.set_option("display.max_columns", None)
om2=pd.get_dummies(om, prefix=['urbanism', 'entry'])
columns=om2.columns
om2.to_clipboard()
array2 = om2.values
X = array2[:, 1:]
Y = array2[:, 0]
print(len(om2.columns))
print("x.shape=",X.shape)
f_selector=mutual_info_regression(X, Y, discrete_features=[4,5,6,8,10,19,20,21,22,23,24,25,26,27,28])
df = pd.DataFrame(f_selector, columns=["1"],index = list(columns)[1:])
for i in range(20):
    dfi = pd.DataFrame({i+2 : mutual_info_regression(X, Y, discrete_features=[4,5,6,8,10,19,20,21,22,23,24,25,26,27,28])},index = list(columns)[1:])
    df=df.join(dfi)
df['avg_score'] = df.mean(axis=1)
df['Rank'] = df['avg_score'].rank(method='dense', ascending=False)
df=df.sort_values("Rank")
df.to_csv(r"data\clustering_results\features_scoring.csv")


# X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=8, step=2)
selector = selector.fit(X, Y)
selector.ranking_
df2 = pd.DataFrame(selector.ranking_, columns=["1"],index = list(columns)[1:])
