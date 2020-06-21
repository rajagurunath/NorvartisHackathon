from dataset import trainDF,testDF
import config
from models import ModelDict,saveModelDict,loadModelDict
from engine import trainModels
from utils import fillMean,addProb
from evaluation import showPerformance
import pandas as pd
def run():
	if Train:
		X=trainDF.iloc[:,2:-1]
		X=fillMean(X)
		
		y=trainDF[config.targetColumn]
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
		X=fillMean(testDF.iloc[:,2:])
		for name,model in trainedModelDict.items():
			allModelPred[name]=model.predict(X)
			allModelPred=addProb(allModelPred,name,model.predict_proba(X))
		allModelPred.to_csv(config.submissionFile,index=False)
	
if __name__=="__main__":
	Train=False
	run()

