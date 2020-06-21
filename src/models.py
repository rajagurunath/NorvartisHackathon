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




