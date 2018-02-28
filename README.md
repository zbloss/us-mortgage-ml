### us-mortgage-ml ###
Training a Linear Regression model to predict Boston housing prices based off of a set of features.

Doing exploratory Data Analysis and Data Visualization on real-world housing data from Boston, MA.

### Steps ###

All work has been done in main.py.
1. import data from scikitlearn
2. format data into a DataFrame
3. perform various visualizations to gain insight
4. train a linear regression model and make Predictions
5. compare the predictions against the actual data with a scatter plot

### Visualizations ###
All Visualizations have been exported to the /plots/ directory
1. ActualVPredMEDV.png
  - Here is the scatterplot comparing the final results of the data
2. BostonHousingCorrHeatmap.png
  - A heatmap showing the correlation each feature has with the Median House value
3. BostonHousingDistPlot
  - a distribution plot using Seaborn of the Median House Value

### Evaluation ###
The Mean Squared Error & Root Mean Squared Error are printed to /data/evals.txt

### DATASET DESCRIPTION ###
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
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

    :Missing Attribute Values: None

    :Creator: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset.
http://archive.ics.uci.edu/ml/datasets/Housing


This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.

The Boston house-price data has been used in many machine learning papers that address regression
problems.   

**References**

   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
   - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)

  ###  ###
