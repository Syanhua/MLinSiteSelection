import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import random
# %matplotlib inline
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
data_matrix = pd.read_csv('data-46-raw.csv', index_col=0)
X = data_matrix.drop('money', axis=1)
Y = data_matrix['money']
# Get column names first
names = X.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_df, columns=names)
#Create a Gaussian Classifier
rf=RandomForestClassifier(n_estimators=100)
#Train the model using the training sets y_pred=clf.predict(X_test)
rf.fit(scaled_df,Y)
importances = list(rf.feature_importances_)
feature_list = X.columns
feature_importances = [(feature, round(importance,2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#[print('Variable: {:50} Importance: {}'.format(*pair)) for pair in feature_importances];
# list of x locations for plotting
x_values = list(range(len(importances)))
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)
useful_feature_index = np.where(cumulative_importances<=0.95)
useful_feature_index = useful_feature_index[0]
X_final = X.iloc[:,0:useful_feature_index[-1]]
# Random Search Training.

# 决策树个数参数取值
n_estimators = list(range(1, 200, 5))

# 最大特征数参数取值
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# 最小叶子结点的参数取值
min_samples_leaf = list(range(1, X_final.shape[0], 10))

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features':max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 500, cv = 3, verbose=2, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_final, Y)




def MachineLearningModel(filename):
    X_test = pd.read_csv(filename, index_col=0)
    predicted = rf_random.best_estimator_.predict(X_test)
    predicted = predicted.tolist()
    return predicted

results = MachineLearningModel('test_new.csv')