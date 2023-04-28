import pandas as pd
#import spacy
#nlp = spacy.load("en_core_web_sm")
#import re
import sklearn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import time
import numpy as np
from sklearn.model_selection import KFold
import statistics

from sklearn.compose import ColumnTransformer
#from sklearn.compose import make_column_transformer
#from sklearn import preprocessing
#from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import f1_score
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import cross_val_score, cross_validate


# DATA PREPROCESSING FUNCTION
# This function takes the product and review data and generates a pandas dataframe
# where each row corresponds  to a product and contains the information needed by 
# the SKlearn pipeline in order to create feature vectors and run a classifier.

# For each product, this function aggregates the review text into a single string, and the 
# summary text into a second string. 

# Each row of the resulting dataframe contains the ASIN ID, 
# the number of reviews received by that product, the fraction of verified reviews,
# the aggregated review text, the aggregated summary text, the mean and standard deviation of 
# the sentiment scores for the reviews, and the mean and standard deviation of the sentiment scores
# for the summaries. Sentiment scores are calculated using the sentiment_analysis helper function.


# inputs: path - a string that has the path to the directory containing the data files
# test: a boolean for whether we are processing the test data (true) or the training data (false)
# outputs: A dataframe that can be passed  to the SKLearn pipeline. This function also saves this
# dataframe as a json file, which can be read so that the preprocessing function does not have to 
# be used every time we run main.py
def data_preprocessing(path, test=False):
    #Create the correct file paths
    if test:
        pfilename = path + "/product_test.json"
        rfilename = path + "/review_test.json" 
    else:
        pfilename = path + "/product_training.json"
        rfilename = path + "/review_training.json"
    
    # get product data
    product_df = pd.read_json(pfilename)
    
    # get review data and remove duplicate reviews
    review_df = pd.read_json(rfilename).drop_duplicates(subset=["reviewerID", "unixReviewTime"], keep="first")
    
    # remove unnecessary columns
    review_df.drop(columns=["reviewerID","vote", "unixReviewTime","reviewTime","style","reviewerName","image"], axis=1 ,inplace=True)
    
    # empty reviews/summaries are filled in with empty strings
    review_df['reviewText'].fillna("", inplace=True)
    review_df['summary'].fillna("", inplace=True)
    
    # review and product dataframes are sorted by ASINs so that their rows correspond to one another
    review_df.sort_values('asin', inplace = True)
    product_df.sort_values('asin', inplace = True)
    
    # group the reviews by ASINs, so that we can iterate through the ASINs in the review data
    group = review_df.groupby("asin")
    
    start_time = time.time()
    
    # this list stores the processed data for each ASIN. This will be converted into a dataframe after
    # it is filled
    datalist = []
    
    # keeps track of the current row in the product dataframe
    count = 0
    
    # iterate through the grouped review data - asin is the current ASIN, and data is 
    # the set of review data corresponding to the current ASIN
    for asin, data in group:
        
        # add up the number of verified  reviews
        verifiedCount = data['verified'].sum()
        
        #add up the number of total reviews
        reviewCount = data['asin'].count()
        
        # calculate the fraction of verified reviews
        percentVerified = verifiedCount / reviewCount
        
        # concatenate the review text into a single string
        reviewText = ' '.join(data['reviewText'])
        
        # concatenate the summary text into a single string
        summaryText = ' '.join(data['summary'])
        
        # get the mean and standard deviation of the sentiment for the reviews
        (rev_mean, rev_stdev) = sentiment_analysis(data['reviewText'])
        
        # get the mean and standard deviation of the sentiment for the summaries
        (sum_mean, sum_stdev) = sentiment_analysis(data['summary'])
        
        # some of the products in the product data do not have reviews, so this 
        # code makes sure that we skip past those products, so that
        # product_df['asin'][count] is always the current ASIN value
        while (product_df['asin'][count] != asin):
               count = count + 1
        
        # with training data, we know the awesomeness value, and we store this in our 
        # output dataframe. If we are processing the test data, then we do not have known
        # awesomeness values
        if test:
            datalist.append([asin,  reviewCount, percentVerified, reviewText, summaryText, rev_mean, rev_stdev, sum_mean, sum_stdev])
            count = count + 1
        else:
            #do train stuff
            awesomeness = product_df['awesomeness'][count]
            datalist.append([asin,  reviewCount, percentVerified, reviewText, summaryText, rev_mean, rev_stdev, sum_mean, sum_stdev, awesomeness])
            count = count + 1
    
    # using the data in datalist, generate the output dataframe with the correct column labels, 
    # based  on whether or not we are processing test data or training data
    if test:
        review_group_df = pd.DataFrame(datalist,columns =['asin', 'numReviews', 'percentVerified','reviewText','summaryText', \
                                                      'reviewMean', 'reviewStDev', 'summaryMean', 'summaryStDev'])    
    else:
        review_group_df = pd.DataFrame(datalist,columns =['asin', 'numReviews', 'percentVerified','reviewText','summaryText', \
                                                          'reviewMean', 'reviewStDev', 'summaryMean', 'summaryStDev', 'awesomeness'])    

    # save the output dataframe to a json file
    if test:
        review_group_df.to_json('cleaned_data_test.json')
    else:
        review_group_df.to_json('cleaned_data.json')
    end_time = time.time()
    print(end_time - start_time)
    
    return review_group_df

#helper function that runs the sentiment analysis
def sentiment_analysis(docs):
    sentiments = []
    
    # the VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # iterate through the current set of documents (either all the reviews or 
    # all the summaries for a given ASIN)
    for doc in docs:
        polarities = sid.polarity_scores(doc)
        
        # The compound polarity is a score between -1 and 1 for the sentiment in the document
        sentiments.append(polarities['compound'])
        
    # if there is only 1 review for a product, then the standard deviation is 0
    if len(sentiments) == 1:
        return (sentiments[0], 0)
    else:
        return (statistics.mean(sentiments) + 1, statistics.stdev(sentiments))
  
# This function creates the classifier pipeline that is trained and then used for predictions
def create_classifier():
    
    # This pipeline runs countvectorizer and TFIDFTransformer to create a vector representation of the TFIDF scores of the words present in a given document
    string_transformer = Pipeline(steps = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())])
    
    # This column transformer runs string_transformer on the aggregated review text and summary text, and combines the resulting vectors in with the other features
    # to generate our feature vectors for each ASIN
    wordbagger = ColumnTransformer(transformers=[("rev", string_transformer, 'reviewText'), ("sum", string_transformer, 'summaryText')], remainder='passthrough')

    # This pipeline combines wordbagger (generating the feature vectors), with 
    # a scaler that scales all the data to be between -1 and 1, 
    # and the Logistic Regression Classifier
    # For logistic  regression, we allow it to run for up to 500 iterations, and ask it to parallelize the operations to reduce compute time
    clf = Pipeline(steps = [("wordbag", wordbagger), ("scale", MaxAbsScaler()), ('classifier',  LogisticRegression(max_iter = 500, solver = "newton-cg", C = 0.1, n_jobs = -1) )])
    
    return clf

# This function trains the model on the entire training dataset
def train_classifier(clf, review_group_df):
    
    # grab only the features that are needed for wordbagger to create feature vectors
    review_features = review_group_df.filter(['numReviews', 'percentVerified', 'reviewText', 'summaryText', 'reviewMean', 'reviewStDev', 'summaryMean', 'summaryStDev'])
    
    # grab the awesomeness results
    y = review_group_df.filter(['awesomeness'])
    
    # fit the model and return the fitted model
    return clf.fit(review_features, np.ravel(y))

# This function runs 10-fold cross-validation on the training dataset    
def cross_validator(clf, review_group_df):
    
    # grab the relevant features
    review_features = review_group_df.filter(['numReviews', 'percentVerified', 'reviewText', 'summaryText', 'reviewMean', 'reviewStDev', 'summaryMean', 'summaryStDev'])
    
    # grab the awesomeness scores
    y = review_group_df.filter(['awesomeness'])
    start = time.time()
    print("I started at " + str(start))
    
    # this runs the k-fold cross-validation automatically, with 10 folds and parallel computing
    cv10_results = cross_validate(clf, review_features, np.ravel(y), cv=10, n_jobs = -1, scoring = ['f1_micro', 'precision', 'recall'])
    end = time.time()
    print("duration: " + str((end - start)/60))
    return cv10_results

#This function takes in the test data and a trained classifier, and predicts the awesomeness scores
# these awesomeness scores are then saved to a dataframe and to the predictions.json output file
def generate_predictions(review_group_df, clf):
    review_features = review_group_df.filter(['numReviews', 'percentVerified', 'reviewText', 'summaryText', 'reviewMean', 'reviewStDev', 'summaryMean', 'summaryStDev'])
    predictions = pd.Series(clf.predict(review_features))
    result_df = pd.DataFrame({'asin': review_group_df['asin'], 'awesomeness': predictions})
    result_df.to_json('predictions.json')
    return result_df
    

#FUNCTION CALLS

# generating the cleaned_data json
#review_group_df_train = data_preprocessing('../devided_dataset_v2/CDs_and_Vinyl/train/', test = False)
#read in the training data
review_group_df_train = pd.read_json('cleaned_data.json')

#make the classifier
clf = create_classifier()

# 10-fold cross-validation on our data
#cross_validation_results = cross_validator(clf, review_group_df_train)
#print(cross_validation_results)

# train classifier
train_classifier(clf, review_group_df_train)

# generate feature vectors from the test data
#review_group_df_test = data_preprocessing('../devided_dataset_v2/CDs_and_Vinyl/test1/', test = True)
review_group_df_test = pd.read_json('cleaned_data_test.json')

# generate predictions and save predictions.json to the current directory
result_df = generate_predictions(review_group_df_test, clf)
