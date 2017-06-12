import seaborn as sns
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics.regression import mean_squared_error
##Reading the data
input_file = "kc_house_data.csv"
df = pd.read_csv(input_file)
#dropping irrelevant features
df.drop(['date'], axis=1)
#plotting using seaborn to visualize relationship
sns.plt.show()
sns.pairplot(df, x_vars=["sqft_living", "floors",'bathrooms', 'bedrooms', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_basement', 'yr_built'], y_vars="price", kind='reg')
sns.plt.show()

X = df[['sqft_living','floors','bathrooms', 'bedrooms', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_basement', 'yr_built']].as_matrix()
Y = df['price'].values
train_feature, test_feature, train_target, test_target = train_test_split(X, Y, random_state=1)
#trying to fit model using linear regression
reg = AdaBoostRegressor()
reg.fit(train_feature, train_target)
#predicting the test data
pred = reg.predict(test_feature)
##calculating mean squared error
msq_error= mean_squared_error(test_target, pred)
print 'mean square error: ' , msq_error, '\n'
##printing score
print 'Regression accuracy score:  ', reg.score(X, Y), '\n'
print 'Below is actual price vs predicted price', '\n'
print (pred), '\n'
print (train_target)