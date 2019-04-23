#Importing the libraries required
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os

path= "D:\work\Big-Mart-Sales-Prediction-master"
filename_train = "\\train\\train.csv"
filename_test = "\\test\\test.csv"
filename_feature = "\\features.csv\\features.csv"
filename_store = "stores.csv"


dataset= pd.read_csv(os.path.realpath(path+filename_train), names=['Store','Dept','Date','weeklySales','isHoliday'],sep=',', header=0)
#test=pd.read_csv(os.path.realpath(path+filename_test))
features=pd.read_csv(os.path.realpath(path+filename_feature),sep=',', header=0, names=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment','IsHoliday']).drop(columns=['IsHoliday'])
stores=pd.read_csv("stores.csv", names=['Store','Type','Size'],sep=',', header=0)
dataset = dataset.merge(stores, how='left').merge(features, how='left')
# determine the training data and testing data
# the given testing data and training data has the store, dept, date
# But we have the features of the store and the date, which are parameters influencing the results and should be taken into the input
# the training data and testing data will combine all thesis features before input into the model
dataset = pd.get_dummies(dataset, columns=["Type"])
dataset[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = dataset[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)
dataset['Month'] = pd.to_datetime(dataset['Date']).dt.month
dataset = dataset.drop(columns=["Date", "CPI", "Fuel_Price", 'Unemployment', 'MarkDown3'])

def extraTreesRegressor():
    clf = ExtraTreesRegressor(n_estimators=100,max_features='auto', verbose=1, n_jobs=1)
    return clf

def predict_(m, test_x):
    return pd.Series(m.predict(test_x))

def model_():
#     return knn()
    return extraTreesRegressor()
#     return svm()
#     return nn()
#     return randomForestRegressor()    

def train_(train_x, train_y):
    m = model_()
    m.fit(train_x, train_y)
    return m

def train_and_predict(train_x, train_y, test_x):
    m = train_(train_x, train_y)
    return predict_(m, test_x), m

def calculate_error(test_y, predicted, weights):
    return mean_absolute_error(test_y, predicted, sample_weight=weights)




kf = KFold(n_splits=5)
splited = []
for name, group in dataset.groupby(["Store", "Dept"]):
    group = group.reset_index(drop=True)
    trains_x = []
    trains_y = []
    tests_x = []
    tests_y = []
    if group.shape[0] <= 5:
        f = np.array(range(5))
        np.random.shuffle(f)
        group['fold'] = f[:group.shape[0]]
        continue
    
    fold = 0
    for train_index, test_index in kf.split(group):
        group.loc[test_index, 'fold'] = fold
        fold += 1
    splited.append(group)


splited = pd.concat(splited).reset_index(drop=True)


best_model = None
error_cv = 0
best_error = np.iinfo(np.int32).max
for fold in range(5):
    dataset_train = splited.loc[splited['fold'] != fold]
    dataset_test = splited.loc[splited['fold'] == fold]
    train_y = dataset_train['weeklySales']
    train_x = dataset_train.drop(columns=['weeklySales', 'fold'])
    test_y = dataset_test['weeklySales']
    test_x = dataset_test.drop(columns=['weeklySales', 'fold'])
    print(dataset_train.shape, dataset_test.shape)
    predicted, model = train_and_predict(train_x, train_y, test_x)
    weights = test_x['isHoliday'].replace(True, 5).replace(False, 1)
    error = calculate_error(test_y, predicted, weights)
    error_cv += error
    print(fold, error)
    if error < best_error:
        print('Find best model')
        best_error = error
        best_model = model
error_cv /= 5


test=pd.read_csv(os.path.realpath(path+filename_test), names=['Store','Dept','Date','isHoliday'],sep=',', header=0)
test=test.merge(stores,how='left').merge(features, how='left')
test = pd.get_dummies(test, columns=["Type"])
test[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = test[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)
test = test.fillna(0)
column_date = test['Date']
test['Month'] = pd.to_datetime(test['Date']).dt.month
test = test.drop(columns=["Date", "CPI", "Fuel_Price", 'Unemployment', 'MarkDown3'])


predicted_test = best_model.predict(test)
test['weeklySales'] = predicted_test
test['Date'] = column_date
test['id'] = test['Store'].astype(str) + '_' + test['Dept'].astype(str) + '_' + test['Date'].astype(str)
test = test[['id', 'weeklySales']]
test = test.rename(columns={'id': 'Id', 'weeklySales': 'Weekly_Sales'})
                
test.to_csv('output.csv', index=False)
