"""
### dataset.py
- This file acts as a virtual DB where the dataset was read and keep it in memory
- Along with the train, test, all the feature Engineering was applied to the dataset and stored in memory to be consumed by all models while training and prediction
- All the meta data related to the dataset, like categorical features, time features etc are also generated by this script


"""
from path import Path 
from sklearn.model_selection import train_test_split
import pandas as pd
import config


DIR=Path(config.TRAINING_FILE)
print(DIR.files())
trainDF=pd.read_csv(DIR+"Train.csv")
testDF=pd.read_csv(DIR+"Test.csv")

if config.useTransform:
	from transforms import add_datepart
	X=add_datepart(trainDF.iloc[:,1:-1],"DATE","date_")
	testDF=add_datepart(testDF,"DATE","date_")
	
else:
	X=trainDF.iloc[:,2:-1]

print("X Train columns :",X.columns)
y=trainDF[config.targetColumn]
X_train, X_validation, y_train, y_validation =\
	 		train_test_split(X, y, train_size=0.8, random_state=1234)

def getCatgoricalFeatures(X,return_index=True):
	cat_features = X.nunique()[X.nunique()<=10].index.tolist()
	collist=X.columns.tolist()
	cat_features_code=[]
	for fe in cat_features:
		cat_features_code.append(collist.index(fe))
	if return_index:
		res=cat_features_code
	else:
		res=cat_features
	return res
