"""
### models.py
- This script prepares a container to hold different model from different framework in memory which is then used by engine to train all the model from the container
- we can also specify to use AutoML from h2o

"""
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import catboost
from catboost import CatBoostClassifier
import config
import pickle

ModelDict={
	'RF':RandomForestClassifier(**config.RFParams),
	"GB":GradientBoostingClassifier(**config.GBParams),
	"catboost":CatBoostClassifier(**config.catboostParams)
}

if config.useAutoML:
	from h2o.automl import H2OAutoML
	ModelDict['h2o']=H2OAutoML(**config.h2oParams)

def saveModelDict(ModelDict):
	pickle.dump(ModelDict,open(config.ModelDictPath,'wb'))

def loadModelDict():
	return pickle.load(open(config.ModelDictPath,'rb'))




