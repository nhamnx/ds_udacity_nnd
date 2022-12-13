import sys
import numpy as np
import pandas as pd 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge 2 datasets from CSV.
    
    Arguments:
        messages_filepath: CSV file path to messages dataset 
        categories_filepath: CSV file pat to categories dataset
    Output:
        df: merged dataframe from messages and categories dataframe
    """
    #Load data from csv files 
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    #Merge 2 dataframes
    df = pd.merge(messages_df, categories_df, how='left', on='id')

    return df



def clean_data(df):
    """
    Clean dataset
    Arguments:
        df: merged dataframe from messages and categories dataframe
    Output:
        df: cleaned input dataframe 
    """
    # Split 'categories' columns into new dataframe with 36 columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[:1]
    categories_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]
    categories.columns = categories_colnames

    # Convert category values 0 and 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
    
    # Replace '2' appeared in 'related' column to '1'
    categories['related'] = categories['related'].replace(to_replace=2, value=1)
    
    # Drop 'categories' column in original dataframe and merge with new categories dataframe
    df = df.drop(columns='categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    #Drop duplicated records
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Store cleaned data to SQLite database
    Arguments:
        df: cleaned dataframe
        database_filename: name of our database
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Disaster_Database', engine, index=False, if_exists='replace')

def main():
    """
    Main function:
        Load and merge 2 datasets from CSV
        Clean dataset
        Store cleaned data to SQLite database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()