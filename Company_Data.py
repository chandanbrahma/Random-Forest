


##importing the dataset
import pandas as pd

data= pd.read_csv('E:\\assignment\\randomforest\\Company_Data.csv')

data.head()
## we do have 11 rows and 400 columns with the target variale 'sales'

data.describe()
data.info()
## we do not have any null values inside the dataset but 3 columns need to be converted into integer format


##finding the unique values of the sales column
data['Sales'].unique()

##converting the sales column in catagorical form and catagorising them
category= pd.cut(data.Sales,bins=[0,3,10,20],labels=['low','medium','high'])

##inserting the column inside the dataframe
data.insert(1,'sales_type',category)

data.sales_type.value_counts()

## droping the sales column
data.drop(columns=['Sales'],inplace= True)

data['ShelveLoc'],_ = pd.factorize(data.ShelveLoc)

data['Urban'],_ = pd.factorize(data.Urban)

data['US'],_ = pd.factorize(data.US)

data['sales_type'],_ = pd.factorize(data.sales_type)

data.head()
data.info()


## model  building
## choosing the predictor and target columns
x=data.iloc[:,1:]

y=data.iloc[:,0]
## choosing sales as the target columns and rest are the input columns


## training and spliting the dataset
from sklearn.model_selection import train_test_split

##applying on x and y
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3)

##importing Random forest and applying on the model
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(random_state=1)
model.fit(x_train, y_train)

# use the model to make predictions with the test data
y_pred = model.predict(x_test)
from sklearn import  metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy 

##we got an accuracy of 80%