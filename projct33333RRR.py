

# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV 
%matplotlib inline
import matplotlib.pyplot as plt
from scipy import signal 
import pickle

from sklearn import utils
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support 

# Importing the dataset
dataframe = pd.read_csv("dataframe_hrv.csv")
dataframe.describe()

dataframe = dataframe.reset_index(drop=True)
print(dataframe.columns)
def missing_values(df):
    df = df.reset_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df[~np.isfinite(df)] = np.nan
   # df.plot( y=["HR"])
    df['HR'].fillna((df['HR'].mean()), inplace=True)
    df['HR'] = signal.medfilt(df['HR'],13) 
   # df.plot( y=["HR"])
    df=df.fillna(df.mean(),inplace=True)
    return df

dataframe = missing_values(dataframe)
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, 23].values

# splitting the dataset into the training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =200, random_state = 0)
regressor.fit(X_train, y_train)


# Predicting a new result
y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((4129,1)).astype(int),values = X, axis = 1)

X_opt =X[:,[0,1,2,4,5,6,8,9,10,12,14,15,16,17,18,19,20,21,22,23]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
print("RESIDUAL ERRORS IN THE PLOT:")
plt.style.use('fivethirtyeight')
#plot the residual errors in training data
plt.scatter(regressor.predict(X_train),regressor.predict(X_train)-y_train,color ="green", s = 12, label = 'TRAIN DATA')
plt.scatter(regressor.predict(X_test),regressor.predict(X_test)-y_test,color ="blue", s = 12, label = 'Test DATA')

plt.hlines(y=0,xmin=0,xmax = 11 , linewidth =2)
#plotting the legend
plt.legend(loc = 'upper right')
plt.title('Residual Errors')
plt.show()



# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X[:,0]), max(X[:,0]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,0], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,0], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('INDEX')
plt.ylabel('stress')
plt.show()

X_grid = np.arange(min(X[:,1]), max(X[:,1]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,1], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,1], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('ECG')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,2]), max(X[:,2]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,2], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,2], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('EMG')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,3]), max(X[:,3]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,3], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,3], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('HR')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,4]), max(X[:,4]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,4], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,4], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('RESP')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,5]), max(X[:,5]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,5], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,5], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('SECONDS')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,6]), max(X[:,6]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,6], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,6], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('FOOTGSR')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,7]), max(X[:,7]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,7], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,7], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('HANDGSR')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,8]), max(X[:,8]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,8], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,8], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('INTERVAL IN SECONDS')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,9]), max(X[:,9]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,9], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,9], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('Marker')
plt.ylabel('stress')
plt.show()

X_grid = np.arange(min(X[:,10]), max(X[:,10]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,10], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,10], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('newtime')
plt.ylabel('stress')
plt.show()
# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X[:,11]), max(X[:,11]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,11], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,11], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('time')
plt.ylabel('stress')
plt.show()

X_grid = np.arange(min(X[:,12]), max(X[:,12]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,12], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,12], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('NNRR')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,13]), max(X[:,13]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,13], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,13], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('AVNN')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,14]), max(X[:,14]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,14], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,14], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('SDNN')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,15]), max(X[:,15]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,15], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,15], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('RMSSD')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,16]), max(X[:,16]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,16], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,16], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('PNN50')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,17]), max(X[:,17]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,17], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,17], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('TP')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,18]), max(X[:,18]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,18], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,18], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('ULf')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,19]), max(X[:,19]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,19], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,19], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('VLF')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,20]), max(X[:,20]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,20], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,20], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('LF')
plt.ylabel('stress')
plt.show()

X_grid = np.arange(min(X[:,21]), max(X[:,21]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,21], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,21], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('HF')
plt.ylabel('stress')
plt.show()
X_grid = np.arange(min(X[:,22]), max(X[:,22]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test[:,22], y_test, color = 'blue',label = "tested set")
plt.scatter(X_test[:,22], y_pred, color = 'green', label = "predicted set")
plt.legend(loc = 'upper right')
plt.title('(Regression Results)')
plt.xlabel('LF_HF')
plt.ylabel('stress')
# Applying k-Fold Cross Validation
"""from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 24)
accuracies.mean()"""


feature_list = ['ECG','EMG','	HR',	'RESP','Seconds','footGSR','handGSR','interval in seconds',	'marker','newtime','time',	'NNRR','AVNN','SDNN','RMSSD','pNN50','TP','ULF','VLF','LF','HF','LF_HF']
imp =  list(regressor.feature_importances_)
feature_importances =[(X,round(importance,2)) for X,importance in zip(feature_list,imp)]
feature_importances = sorted(feature_importances,key = lambda x: x[1],reverse = True)
x_values = list(range(len(imp)))
plt.bar(x_values,imp,orientation = 'vertical',color = 'r',edgecolor = 'k',linewidth = 1.2)
plt.xticks(x_values,feature_list,rotation = 'vertical')
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable importances')
plt.show()

from sklearn.metrics import explained_variance_score
print("explained_variance_score :",explained_variance_score(y_test,y_pred))#closer the value near to 1 higher the accuracy of the data
from sklearn.metrics import mean_absolute_error
print("mean_absolute_error :",mean_absolute_error(y_test,y_pred))#lesser the value higher will be the accuracy
from sklearn.metrics import mean_squared_error
print("mean_squared_error: ",mean_squared_error(y_test,y_pred))
from sklearn.metrics import median_absolute_error
print("median_absolute_error: ",median_absolute_error(y_test,y_pred))
from sklearn.metrics import r2_score
print("r2_score : ",r2_score(y_test,y_pred, multioutput =None))
confidence = regressor.score(X_test, y_test)
print('confidence:', confidence)
acc = confidence * 100
print ("percentage of  accuracy is:",acc)

sorted_importances = [imp[1] for importance in feature_importances]

sorted_features = [imp[0] for importance in feature_importances]
cumulative_importances = np.cumsum(sorted_importances)
plt.hlines(y=0.95,xmin=0,xmax = len(sorted_importances),color = 'r',linestyles = 'dashed')
plt.xticks(x_values, sorted_features, rotation = 'vertical')
plt.xlabel('Variable')
plt.ylabel('Cummulative importance')
plt.title('Cummulative importances')
plt.plot(x_values,cumulative_importances,'g-')

plt.show()
"""used in classification 
from sklearn.metrics import hamming_loss
hamming_loss(y_test,y_pred)

from sklearn.metrics import log_loss
log_loss(y_test,y_pred)
from sklearn.metrics import zero_one_loss
zero_one_loss(y_test,y_pred)
from sklearn.metrics import brier_score_loss
brier_score_loss(y_test,y_pred){worked}"""
"""errors = abs(y_pred - y_test)
print("error:",errors)
#print the mean absolute error
mean_absolute_error = np.mean((y_pred - y_test)**2)
print("mean absolute error :",mean_absolute_error)
#mean absolute percentage error(Mape)
mape = 100*(errors/y_test)
print("mean absolute %age error :",mape)
mean_ab=np.mean(mape)

print("mean absolute %age error :",mean_ab)"""
"""# calculate and diplay the accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:',accuracy)"""