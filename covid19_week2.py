import numpy as np
import pandas as pd

train = pd.read_csv("train2.csv")
test = pd.read_csv("test2.csv")
submission = pd.read_csv("submission02.csv")
#print(train.head())
#print( )

#train["Date"] = train["Date"].apply(lambda x: x.replace("/",""))
#train["Date"]  = train["Date"].astype(int)
#train.tail()

train = train.drop(['Province_State'],axis=1)
train = train.dropna()
#print(train.isnull().sum())
#print( )

#test["Date"] = test["Date"].apply(lambda x: x.replace("/",""))
#test["Date"]  = test["Date"].astype(int)
#print(test.isnull().sum())
#print( )

x = train[['Lat', 'Long', 'Date']]
y1 = train[['Confirmed_Cases']]
y01 = np.ravel(y1)
y2 = train[['Fatalities']]
y02 = np.ravel(y2)
x_test = test[['Lat', 'Long', 'Date']]

from sklearn.ensemble import RandomForestClassifier
Tree_model = RandomForestClassifier(max_depth=200, random_state=0)

Tree_model.fit(x,y01)
pred1 = Tree_model.predict(x_test)
pred1 = pd.DataFrame(pred1)
pred1.columns = ["ConfirmedCases_prediction"]
#print( )
#print(pred1.head())
#print( )

Tree_model.fit(x,y02)
pred2 = Tree_model.predict(x_test)
pred2 = pd.DataFrame(pred2)
pred2.columns = ["Death_prediction"]

Sub = pd.read_csv("submission02.csv")
sub_new = Sub[["Forecast_Id"]]
sub_new

submit = pd.concat([pred1,pred2,sub_new],axis=1)
#print( )
#print(submit.head())
#print( )


submit.columns = ['Confirmed_Cases', 'Fatalities', 'Forecast_Id']
submit = submit[['Forecast_Id','Confirmed_Cases', 'Fatalities']]

submit["Confirmed_Cases"] = submit["Confirmed_Cases"].astype(int)
submit["Fatalities"] = submit["Fatalities"].astype(int)

print( )
print(submit.describe())
print( )

Sub = submit
Sub.to_csv('results2.csv', index=False)