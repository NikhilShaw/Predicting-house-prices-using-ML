import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

## accessing data
input_file = "kc_house_data.csv"
df = pd.read_csv(input_file)
## dropping redundant data
df.drop(['id','date'], axis=1)
sns.plt.show()
## plotiing using seaborn
sns.pairplot(df, x_vars=["sqft_living"], y_vars="price", kind='reg')
sns.plt.show()

X = df['sqft_living']
Y = df['price']
## splitting data for training and testing of data
train_feature, test_feature, train_target, test_target = train_test_split(X, Y, random_state=1)
reg = LinearRegression()
##fitting the model
reg.fit(train_feature, train_target)
## predicting prices for training set 
pred = reg.predict(train_feature)
## printing the accuracy of the model
print accuracy_score(train_target, pred)

