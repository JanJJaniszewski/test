#########
# Imports
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge, RidgeClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer, OneHotEncoder, LabelEncoder
from sklearn.svm import SVR

import Functions.General as fg
import config_main as cm

# Seed and measurement
seed = 7
scoring = 'accuracy' #r2
cv=5

##################
# Classifier model
r_dummy = dict(clf=[DummyClassifier()],
               clf__strategy=['most_frequent'])

r_lasso = dict(clf=[Lasso()],
               clf__alpha=[0.1, 1.0, 10.0, 100.0],
               clf__random_state=[seed])

r_logistic = dict(clf=[LogisticRegression()],
               clf__random_state=[seed])

r_ridge = dict(clf=[RidgeClassifier()],
               clf__alpha=[0.1, 1.0, 10.0, 100.0],
               clf__random_state=[seed])

r_forest = dict(clf=[RandomForestClassifier()],
                clf__n_estimators=[500],
                #clf__criterion=['mae'],
                clf__max_depth=[3, 5, None],
                clf__min_samples_leaf=[5, 10],
                clf__random_state=[seed],
                clf__max_features=[3, None])

r_svm = dict(clf=[SVR()],
             clf__C=[0.1, 1.0, 10.0, 100.0, 200.0, 500.0])

##########
# Pipeline
# sc = pyspark.SparkContext("local", "Simple App")
pipe = Pipeline([
    ('onehotencoder', OneHotEncoder(categorical_features=[1, 6], sparse=False)),
    ('standardize', StandardScaler()),
    #('select', SelectFromModel(ExtraTreesRegressor(), threshold='median')),
    ('clf', DummyClassifier())])
param_grid = [r_dummy, r_forest, r_logistic]
grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1)

# Log model
logger = fg.create_logger('Model', cm.model_log)
logger.info(cm.currentVersion)
logger.info(grid_search)

if __name__ == '__main__':
    pass
