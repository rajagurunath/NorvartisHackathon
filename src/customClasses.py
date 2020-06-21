"""
### customClasses.py
- To implement voting classifier from all the predictions(from csv) made by the model 
- sklearn doesn't support to build voting classifier from the csv predictions, so Custom Classifier like `FromFileClassifier` (sklean estimator) and `WrapVotingClassifier` 
- `FromFileClassifier` reads the relavant model csv and acts as a Dummy classifier which just throws the predictions from the CSV
- `WrapVotingClassifier` builds the soft voting from the given list of Fileclassifiers
- HardVoting can be performed using a separated function `hardVoting`

"""
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.ensemble._voting import _BaseVoting
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.multiclass import check_classification_targets
#from ..utils.validation import _deprecate_positional_args
import pandas as pd
import numpy as np

class FromFileClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,filePath,columnName='RF'):
        super(FromFileClassifier,self).__init__()
        self.filePath=filePath
        self.columnName=columnName
        self.data=pd.read_csv(self.filePath).astype(float)
    def fit(self,X,Y):
        return self
    def predict(self,X):
        """
        X is dummy here
        """
        predCols=self.data.columns[self.data.columns.str.startswith(self.columnName)]
        print(predCols)
        #preds.append(self.data[predCols[0]])
        #print(self.data[predCols[0]].values)
        return self.data[predCols[0]].values
    
    def predict_proba(self,X):
        predCols=self.data.columns[self.data.columns.str.startswith(self.columnName)]
        #probhas.append(self.data[predCols[1:]].values)
        #print(self.data[predCols[1:]].values)
        return self.data[predCols[1:]].values
    
class WrapVotingClassifier(ClassifierMixin, _BaseVoting):
    
    @_deprecate_positional_args
    def __init__(self, estimators, *, voting='hard', weights=None,
                 n_jobs=None, flatten_transform=True, verbose=False):
        super().__init__(estimators=estimators)
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.
            .. versionadded:: 0.18
        Returns
        -------
        self : object
        """
        check_classification_targets(y)
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        #self.le_ = LabelEncoder().fit(y)
        #self.classes_ = self.le_.classes_
        #transformed_y = self.le_.transform(y)
        self.estimators_=[est[1] for est in self.estimators]
        return self #super().fit(X, transformed_y, sample_weight)

    def predict(self, X):
        """Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        maj : array-like of shape (n_samples,)
            Predicted class labels.
        """
        #check_is_fitted(self)
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x, weights=None)),
                axis=1, arr=predictions)

        #maj = self.le_.inverse_transform(maj)

        return maj

    def _collect_probas(self, X):
        """Collect results from clf.predict calls."""
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting."""
        #check_is_fitted(self)
        avg = np.average(self._collect_probas(X), axis=0,
                         weights=self._weights_not_none)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        avg : array-like of shape (n_samples, n_classes)
            Weighted average probability for each class per sample.
        """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        return self._predict_proba

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        probabilities_or_labels
            If `voting='soft'` and `flatten_transform=True`:
                returns ndarray of shape (n_classifiers, n_samples *
                n_classes), being class probabilities calculated by each
                classifier.
            If `voting='soft' and `flatten_transform=False`:
                ndarray of shape (n_classifiers, n_samples, n_classes)
            If `voting='hard'`:
                ndarray of shape (n_samples, n_classifiers), being
                class labels predicted by each classifier.
        """
        #check_is_fitted(self)

        if self.voting == 'soft':
            probas = self._collect_probas(X)
            if not self.flatten_transform:
                return probas
            return np.hstack(probas)

        else:
            return self._predict(X)
def hardVoting(x):
    s=x.sum()
    ones=s
    zeros=4-ones
    if zeros>ones:
        res=0
    else:
        res=1
    return res
if __name__=='__main__':
	clfs=[(clf,FromFileClassifier("models_pred.csv",columnName=clf)) for clf in ["RF","GB","catboost","h2o"]]
	vclf=WrapVotingClassifier(estimators=clfs,voting='soft')
	vclf.fit(train.iloc[:,1:].values,train.iloc[:,-1].values)
	votePred=vclf.predict(train.iloc[:,1:])