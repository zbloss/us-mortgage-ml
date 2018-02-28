import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import classification_report, mean_squared_error
from pylab import savefig

# Setting the project path. This will be used to save various plots


# Using real-world data from scikitlearns load_boston dataset
boston = load_boston()
print(boston['DESCR'])
boston.keys()
# Creating a Pandas DataFrame from the boston data and feature_names in order
# to more easily maniupalte the data.
boston_df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
target_df = pd.DataFrame(boston['target'], columns=['MEDV'])
boston_df = pd.concat([boston_df, target_df], axis=1)

# Checking for NaN values
boston_df.isnull().values.any()

boston_df.info()
boston_df.describe()

'''
Boston House Prices dataset
===========================

Notes
------
Data Set Characteristics:

    :Number of Instances: 506

    :Number of Attributes: 13 numeric/categorical predictive

    :Median Value (attribute 14) is usually the target

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centrescentres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
'''
plots_path = '/home/zbloss/Github/us-mortgage-ml/plots/'
data_path = '/home/zbloss/Github/us-mortgage-ml/data/'


boston_distplot = sns.distplot(boston_df['MEDV'])
boston_distplot.get_figure().savefig(plots_path + "BostonHousingDistPlot.jpg")

sns.distplot(boston_df['MEDV'])

bdf_corr = boston_df.corr()

# Here we see which features have the strongest correlation on Median value
bdf_corr['MEDV'].sort_values(ascending=False)
# We can see the factor with the largest correlation is number of rooms

plt.figure(figsize=(10,10))
sns.heatmap(boston_df.corr(), annot=True, cmap='coolwarm')

plt.figure(figsize=(10,10))
sns.heatmap(boston_df.corr(), annot=True, cmap='coolwarm').get_figure().savefig(plots_path + 'BostonHousingCorrHeatmap.png')

# Exporting to excel spreadsheet
boston_df.to_excel(data_path + 'boston_df.xls')

# Now the train and test variables are separated
X = boston_df.drop('MEDV', axis=1)
y = boston_df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model gets instantiated and trained on the training data
lrm = LinearRegression()
lrm.fit(X_train, y_train)

# predictions are made and compared to the test data
preds = lrm.predict(X_test)


print(lrm.intercept_)
print(lrm.coef_)

plt.figure(figsize=(16,8))
plt.scatter(y_test, preds)
plt.xlabel('Actual Housing Data')
plt.ylabel('Predicted Housing Data')
plt.title('Actual vs Predicted Median Housing Values')
savefig(plots_path + 'ActualVPredMEDV.png')


### Calculating MSE ###

f = open(data_path + "evals.txt", "w")



MSE = mean_squared_error(y_test, preds)
RMSE = np.sqrt(mean_squared_error(y_test, preds))
print("The MSE is {} \nThe RMSE is {}".format(MSE, RMSE))
f.write("The MSE is {} \nThe RMSE is {}".format(MSE, RMSE))
f.close()
