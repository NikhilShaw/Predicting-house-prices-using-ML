
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics.regression import mean_squared_error
##Reading the data
input_file = "kc_house_data.csv"
df = pd.read_csv(input_file)
#dropping irrelevant features
df.drop(['id','date'], axis=1)
#plotting using seaborn to visualize relationship
sns.plt.show()
sns.pairplot(df, x_vars=["sqft_living"], y_vars="price", kind='reg')
sns.plt.show()
print df['sqft_living'].shape
print df['price'].shape

X = df[['sqft_living']]
Y = df[['price']]
train_feature, test_feature, train_target, test_target = train_test_split(X, Y, random_state=1)
#trying to fit model using linear regression
reg = LinearRegression()
reg.fit(train_feature, train_target)
#predicting the test data
pred = reg.predict(train_feature)
##calculating mean squared error
print mean_squared_error(train_target, pred)

