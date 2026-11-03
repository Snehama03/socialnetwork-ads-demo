import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv("Social_Network_Ads.csv")

x=df.drop(['User ID','Purchased'],axis=1)
y=df['Purchased']

x=pd.get_dummies(x,columns=['Gender'],dtype=int)
feature_columns=x.columns.tolist() 

x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42
)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print('Accuracy=',accuracy_score(y_test,y_pred))

#save the model
pickle.dump(model,open('model.pkl','wb'))

#save scaler
pickle.dump(scaler,open('scaler.pkl','wb'))

#save feature columns
pickle.dump(feature_columns,open('feature_columns.pkl','wb'))

print('Model training completed')