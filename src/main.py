from dataset import trainDF,testDF,X,y
import config
from models import ModelDict,saveModelDict,loadModelDict
from engine import trainModels
from utils import fillMean,addProb
from evaluation import showPerformance
import pandas as pd
from transforms import add_datepart
def run(X,y):
	if Train:
		trainedModel=trainModels(ModelDict,X,y)
		trainedModelDict={}
		i=0
		for name in ModelDict.keys():
			trainedModelDict[name]=trainedModel[i]
			i+=1
		showPerformance(trainedModelDict)
		saveModelDict(trainedModelDict)
	else:
		trainedModelDict=loadModelDict()
		allModelPred=pd.DataFrame()
		# if config.useTransform:
		# 	testDF=add_datepart
		X=fillMean(testDF.iloc[:,2:])
		allModelPred[testDF.iloc[:,0].name]=testDF.iloc[:,0]
		for name,model in trainedModelDict.items():
			allModelPred[name]=model.predict(X)
			allModelPred=addProb(allModelPred,name,model.predict_proba(X))
		allModelPred.to_csv(config.submissionFile,index=False)
	
if __name__=="__main__":
	X=fillMean(X)
	y=trainDF[config.targetColumn]
	Train=True
	run(X,y)

