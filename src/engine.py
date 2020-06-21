"""
### engine.py
- This File acts as an engine where training the models from different frameworks will be performed, Thanks to the sklearn abstraction which is implemented by all the frameworks.
- All the models we configured, will run parallely using Jobllib package utilizing the all the cores in the machine
- parallelism code was inspired (taken) from sklearn core package

"""

from models import ModelDict
from joblib import delayed,Parallel
from sklearn.utils import _print_elapsed_time
from dataset import X_validation, y_validation,getCatgoricalFeatures



def _log_message(name, idx, total,verbose=True):
	if verbose:
		return None
	return '(%d of %d) Processing %s' % (idx, total, name)


def _fit_single_estimator(estimator, X, y, sample_weight=None,
                          message_clsname=None, message=None):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        try:
            with _print_elapsed_time(message_clsname, message):
                estimator.fit(X, y, sample_weight=sample_weight)
        except TypeError as exc:
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise TypeError(
                    "Underlying estimator {} does not support sample weights."
                    .format(estimator.__class__.__name__)
                ) from exc
            raise
    elif message_clsname=="catboost":
	    kwargs={}
	    kwargs['X']=X
	    kwargs['y']=y
	    kwargs['cat_features']=getCatgoricalFeatures(X_validation)
	    kwargs['eval_set']=(X_validation,y_validation)
	    estimator.fit(**kwargs)
    else:
        with _print_elapsed_time(message_clsname, message):
            estimator.fit(X, y)
    return estimator
# def _fit_single_estimator(estimator, X, y, sample_weight=None,
#                           message_clsname=None, message=None):
# 	if message_clsname=="catboost":
# 		kwargs={}
# 		kwargs['X']=X
# 		kwargs['y']=y
# 		kwargs['cat_features']=getCatgoricalFeatures(X_validation)
# 		kwargs['eval_set']=(X_validation,y_validation)
	
#     with _print_elapsed_time(message_clsname, message):
# 		    estimator.fit(**kwargs)
# 		    # estimator.fit(X,y)
# 		# with _print_elapsed_time(message_clsname, message):
# 		# 	estimator.fit(X, y)
#     return estimator



def trainModels(ModelDict,X,y,sample_weight=None):
	n_jobs=len(ModelDict)
	estimators_ = Parallel(n_jobs=n_jobs)(
					delayed(_fit_single_estimator)(
							clf, X, y,
							sample_weight=sample_weight,
							message_clsname=name,
							message=_log_message(name,
													idx + 1, len(ModelDict))
					)
					for idx, (name,clf) in enumerate(ModelDict.items())
								 if clf not in (None, 'drop')
				)
	return estimators_


