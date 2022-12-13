import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Load dataset from SQLite database
    Arguments:
        database_filepath: path to SQLite database
    Output:
    X: Features dataframe
    Y: Target dataframe
    category_names: List of category names
    """
    #Load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Disaster_Database', con=engine)
    
    #Split into Feature and Target seperated dataframes
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize text messages with nltk package
    Arguments:
        text: text messages 
    Output:
        clean_tokens: list of cleaned tokens from text input
    """

    # Tokenize text
    tokens = nltk.tokenize.word_tokenize(text)
    # Lemmatize text
    lemmatizer = nltk.WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens

def build_model():
    """
    Build and tune classifier
    Output:
        cv: Classifier
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {'clf__estimator__n_estimators': [10, 20, 40]}
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 3)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of classifier
    Arguments:
        model: Classifier
        X_test: Test Feature dataframe
        Y_test: Test Target dataframe 
        category_names: List of category names
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """
    Save classifier weight as pickle file
    Arguments:
        model: Classifier
        model_filepath: Saved model file path
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()