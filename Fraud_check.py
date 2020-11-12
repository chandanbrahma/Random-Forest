###importing the dataset
import pandas as pd

data = pd.read_csv("E:\\assignment\\randomforest\\Fraud_Check.csv")
data.head()
data.describe()
data.info()

##lets convert all  the non catagory columns into catagory format and replacing them into the dataframe
##catagorising the income column as >30000 is good and less than 30000 as risky income
data['Taxable.Income'].describe()
income_category= pd.cut(data['Taxable.Income'],bins=[0,30000,100000],labels=['risky','good'])

##inserting the column
data.insert(0,'taxable_income',income_category)
data.taxable_income.value_counts()


## now droping the catagorisd columns
data.drop(columns=['Taxable.Income'],inplace=True)
data.info()

##as we have created a dataframe that contains only catagorical variables.So now lets convert the strings into respective intcodes using pandas.
data['taxable_income'],_ = pd.factorize(data['taxable_income'])
data['Marital.Status'],_ = pd.factorize(data['Marital.Status'])
data['Undergrad'],_ = pd.factorize(data['Undergrad'])
data['Urban'],_ = pd.factorize(data['Urban'])

data.head()
data.info()##everything are converted into integer form

##normalizing the data
data_new = (data-data.min())/(data.max()-data.min())
data_new.describe()
###selecting the target as y and predictor as x
x=data_new.iloc[:,1:]
y=data_new.iloc[:,0]

##spliting data randomly into 80% training and 20% test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

##importing Random forest and applying on the model
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(random_state=1)
model.fit(x_train, y_train)

# use the model to make predictions with the test data
y_pred = model.predict(x_test)
from sklearn import  metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy 

## so we got an accuracy of 75%.
