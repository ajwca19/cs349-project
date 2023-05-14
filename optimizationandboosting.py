import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
import re
import sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import time
import numpy as np
from sklearn.model_selection import KFold
import statistics
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import f1_score
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold

#function that runs hyperparameter optimization on given classifiers, taking in pre-processed features (i.e. already transformed)
def hyperparam(features, y):
    rfc_param_grid={'criterion':['gini', 'entropy'],'max_depth':[50,75,100,200,400],'max_features':['sqrt','log2']}
    rfc_clf = RandomForestClassifier(n_jobs=-1)
    rfc_grid_clf = GridSearchCV(rfc_clf, rfc_param_grid, n_jobs=-1)
    start = time.time()
    print("I'm starting Random Forest search at time! " + str(start))
    rfc_grid_clf.fit(features, np.ravel(y))
    end = time.time()
    print("Finished! RF Grid Search took " + str((end - start)/60))

    nb_param_grid={'alpha':[0.01, 0.1, 0.5, 1, 5, 10], 'fit_prior':[True, False]}
    nb_clf = MultinomialNB()
    nb_grid_clf = GridSearchCV(nb_clf, nb_param_grid, n_jobs=-1)
    start = time.time()
    print("I'm starting Multinomial Naive Bayes search!")
    nb_grid_clf.fit(features, np.ravel(y))
    end = time.time()
    print("Finished! NB Grid Search took " + str((end - start)/60))
    
    log_param_grid={'C':[0.01, 0.02, 0.05, 0.1, 0.2, 0.5]}
    log_clf = LogisticRegression(n_jobs=-1)
    log_grid_clf = GridSearchCV(log_clf, log_param_grid, n_jobs=-1)
    start = time.time()
    print("I'm starting Logistic Regression search at time! " + str(start))
    log_grid_clf.fit(features, np.ravel(y))
    end = time.time()
    print("Finished! LR Grid Search took " + str((end - start)/60))
    
    return (pd.DataFrame.from_dict(rfc_grid_clf.cv_results_), pd.DataFrame.from_dict(nb_grid_clf.cv_results_), pd.DataFrame.from_dict(log_grid_clf.cv_results_))
    
#runs adaboost on the logistic regression classifier, taking in pre-processed features. Prints out mean and SD of F1, precision, and recall
def adaboost(features, y):
    just_clf = LogisticRegression(solver="newton-cg", C=0.05, max_iter = 500, n_jobs=-1)
    boosted_clf = AdaBoostClassifier(estimator=just_clf, learning_rate = 0.01, n_estimators=500)
    start = time.time()
    print("I started at " + str(start))
    cv10_results = cross_validate(boosted_clf, features, np.ravel(y), cv=10, n_jobs = -1, scoring = ['f1_macro', 'f1_micro','precision', 'recall'])
    end = time.time()
    print((end - start)/60)
    print(statistics.mean(cv10_results['test_f1_macro']))
    print(statistics.stdev(cv10_results['test_f1_macro']))
    print(statistics.mean(cv10_results['test_precision']))
    print(statistics.stdev(cv10_results['test_precision']))
    print(statistics.mean(cv10_results['test_recall']))
    print(statistics.stdev(cv10_results['test_recall']))
    
#runs gradient boost on the logistic regression classifier, taking in pre-processed features. Prints out mean and SD of F1, precision, and recall
def gradientboost(features, y):
    just_clf = LogisticRegression(solver="newton-cg", C=0.05, max_iter = 500, n_jobs=-1)
    boosted_clf = GradientBoostingClassifier(init=just_clf, learning_rate = 0.01, n_estimators=500)
    start = time.time()
    print("I started at " + str(start))
    cv10_results = cross_validate(boosted_clf, features, np.ravel(y), cv=10, n_jobs = -1, scoring = ['f1_macro', 'f1_micro','precision', 'recall'])
    end = time.time()
    print((end - start)/60)
    print(statistics.mean(cv10_results['test_f1_macro']))
    print(statistics.stdev(cv10_results['test_f1_macro']))
    print(statistics.mean(cv10_results['test_precision']))
    print(statistics.stdev(cv10_results['test_precision']))
    print(statistics.mean(cv10_results['test_recall']))
    print(statistics.stdev(cv10_results['test_recall']))