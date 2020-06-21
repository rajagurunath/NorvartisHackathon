"""
### evaluation.py
- This is nothing but small evaluation before preparing the submission file or before serializing the trained model
- recall and confusion matrix will be printed while training
- Needs to some extra evalution in the future

"""
from sklearn import metrics
from utils import fillMean
from dataset import X_validation,y_validation
X_validation=fillMean(X_validation)

def showPerformance(ModelDict):
	for name,model in ModelDict.items():
		pred=model.predict(X_validation)
		print(name)
		print("="*80)
		print(f"recall score: {metrics.recall_score(y_validation,pred)}")
		print(f"confusion matrix: {metrics.confusion_matrix(y_validation,pred)}")
