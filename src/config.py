# common File configs
TRAINING_FILE=r"../notebooks/Dataset/"
H2oModelFile="../h2o"
ModelDictPath="../model_dict.pkl"
submissionFile="../submission.csv"

# compute related configs
useAutoML=False
targetColumn="MULTIPLE_OFFENSE"
useTransform=True


# Model Hyperparamaters
#RF
RFParams={}
RFParams['n_estimators']=100
RFParams['criterion']="gini"
RFParams['max_depth']=5

#GB
GBParams={}
GBParams['n_estimators']=100
GBParams['learning_rate']=0.1

#catboost
catboostParams = {}
catboostParams['loss_function'] = 'Logloss'
catboostParams['iterations'] = 80
catboostParams['custom_loss'] = 'Recall'#    custom_loss=['AUC',"Recall", 'Accuracy']
catboostParams['random_seed'] = 63
catboostParams['learning_rate'] = 0.5
catboostParams['train_dir']='../catboost'
# h2o
h2oParams={}
h2oParams['max_models']=20
h2oParams['seed']=1
h2oParams['export_checkpoints_dir']=TRAINING_FILE
h2oParams['max_runtime_secs']=900 # 15 minutes




