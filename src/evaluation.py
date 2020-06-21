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
