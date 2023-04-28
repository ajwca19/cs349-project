import pandas as pd
#import spacy
#nlp = spacy.load("en_core_web_sm")
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


#take in a directory path (as string), return a processed dataframe
def data_preprocessing(path, test=False):
    #start_time = time.time()
    
    #create appropriate file path
    if test == False:
        pfilename = path + "/product_training.json"
        rfilename = path + "/review_training.json"
    else:
        pfilename = path + "/product_test.json"
        rfilename = path + "/review_test.json"
    
    #extract files as pandas dataframes
    product_df = pd.read_json(pfilename)
    
    review_df = pd.read_json(rfilename).drop_duplicates(subset=["reviewerID", "unixReviewTime"], keep="first")
    ## 11.66 seconds to get to here
    
    review_df.drop(columns=["reviewerID","vote", "unixReviewTime","reviewTime","style","reviewerName","image"], axis=1 ,inplace=True)
    
    review_df['reviewText'].fillna("", inplace=True)
    review_df['summary'].fillna("", inplace=True)
    
    review_df.sort_values('asin', inplace = True)
    product_df.sort_values('asin', inplace = True)
    
    group = review_df.groupby("asin")
    
    # about the same amount of time to get to here
    start_time = time.time()
    datalist = []
    count = 0
    #awesome_pos = 0
    for asin, data in group:
        verifiedCount = data['verified'].sum()
        reviewCount = data['asin'].count()
        percentVerified = verifiedCount / reviewCount
        if count == 0:
            print(type(data['reviewText']))
        reviewText = ' '.join(data['reviewText'])
        #reviewText = ' '.join(transform_document(x) for x in data['reviewText'])
        #summaryText = ""
        summaryText = ' '.join(data['summary'])
        #summaryText = ' '.join(transform_document(x) for x in data['summary'])
        #reviewText = transform_document(' '.join(data['reviewText']))
        #summaryText = transform_document(' '.join(data['summary']))
        #awesomeness = 0
        
        #SENTIMENT ANALYSIS CHUNK
        (rev_mean, rev_stdev) = sentiment_analysis(data['reviewText'])
        (sum_mean, sum_stdev) = sentiment_analysis(data['summary'])
        while (product_df['asin'][count] != asin):
               count = count + 1
        
        if test:
            #do test stuff
            datalist.append([asin,  reviewCount, percentVerified, reviewText, summaryText, rev_mean, rev_stdev, sum_mean, sum_stdev])
            count = count + 1
        else:
            #do train stuff
            awesomeness = product_df['awesomeness'][count]
            datalist.append([asin,  reviewCount, percentVerified, reviewText, summaryText, rev_mean, rev_stdev, sum_mean, sum_stdev, awesomeness])
            count = count + 1
        
        '''new_row = {'asin': asin, 
                   'numReviews': reviewCount, 
                   'percentVerified': percentVerified, 
                   'reviewText': transform_document(' '.join(data['reviewText'])), 
                   'summaryText': transform_document(' '.join(data['summary'])), 
                   'awesomeness': product_df.loc[product_df['asin'] == asin, 'awesomeness'].values[0]} 
        review_group_df = review_group_df.append(new_row, ignore_index = True)
         '''
         
    if test:
        review_group_df = pd.DataFrame(datalist,columns =['asin', 'numReviews', 'percentVerified','reviewText','summaryText', \
                                                      'reviewMean', 'reviewStDev', 'summaryMean', 'summaryStDev'])    
    else:
        review_group_df = pd.DataFrame(datalist,columns =['asin', 'numReviews', 'percentVerified','reviewText','summaryText', \
                                                          'reviewMean', 'reviewStDev', 'summaryMean', 'summaryStDev', 'awesomeness'])    

    if test:
        review_group_df.to_json('cleaned_data_test.json')
    else:
        review_group_df.to_json('cleaned_data.json')
    end_time = time.time()
    print(end_time - start_time)
    
    return review_group_df

#runs the sentiment analysis
def sentiment_analysis(docs):
    sentiments = []
    sid = SentimentIntensityAnalyzer()
    for doc in docs:
        polarities = sid.polarity_scores(doc)
        sentiments.append(polarities['compound'])
    if len(sentiments) == 1:
        return (sentiments[0], 0)
    else:
        return (statistics.mean(sentiments) + 1, statistics.stdev(sentiments))
    
def create_classifier():
    string_transformer = Pipeline(steps = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())])
    wordbagger = ColumnTransformer(transformers=[("rev", string_transformer, 'reviewText'), ("sum", string_transformer, 'summaryText')], remainder='passthrough')

    clf = Pipeline(steps = [("wordbag", wordbagger), ("scale", MaxAbsScaler()), ('classifier',  LogisticRegression(max_iter = 500, n_jobs = -1) )])
    
    return clf

def train_classifier(clf, review_group_df):
    review_features = review_group_df.filter(['numReviews', 'percentVerified', 'reviewText', 'summaryText', 'reviewMean', 'reviewStDev', 'summaryMean', 'summaryStDev'])
    y = review_group_df.filter(['awesomeness'])
    return clf.fit(review_features, np.ravel(y))
    
def cross_validate(clf, review_group_df):
    review_features = review_group_df.filter(['numReviews', 'percentVerified', 'reviewText', 'summaryText', 'reviewMean', 'reviewStDev', 'summaryMean', 'summaryStDev'])
    y = review_group_df.filter(['awesomeness'])
    start = time.time()
    print("I started at " + str(start))
    # this runs the k-fold cross-validation automatically?
    cv10_results = cross_val_score(clf, review_features, np.ravel(y), cv=10, n_jobs = -1, scoring = 'f1_macro')
    end = time.time()
    print("duration: " + str((end - start)/60))
    return cv10_results

#take in a trained classifier and test data
def generate_predictions(review_group_df, clf):
    review_features = review_group_df.filter(['numReviews', 'percentVerified', 'reviewText', 'summaryText', 'reviewMean', 'reviewStDev', 'summaryMean', 'summaryStDev'])
    predictions = pd.Series(clf.predict(review_features))
    result_df = pd.DataFrame({'asin': review_group_df['asin'], 'awesomeness': predictions})
    result_df.to_json('predictions.json')
    return result_df
    
## Actually training the model and running it on the test data

# generating the cleaned_data json
#review_group_df_train = data_preprocessing('../devided_dataset_v2/CDs_and_Vinyl/train/', test = False)
#read in the training data
review_group_df_train = pd.read_json('cleaned_data.json')

#make the classifier
clf = create_classifier()

# 10-fold cross-validation on our data
#cross_validation_results = cross_validate(clf, review_group_df_train)
#print(cross_validation_results)

# train classifier
train_classifier(clf, review_group_df_train)

# generate feature vectors from the test data
#review_group_df_test = data_preprocessing('../devided_dataset_v2/CDs_and_Vinyl/test1/', test = True)
review_group_df_test = pd.read_json('cleaned_data_test.json')

# generate predictions and save predictions.json to the current directory
result_df = generate_predictions(review_group_df_test, clf)
