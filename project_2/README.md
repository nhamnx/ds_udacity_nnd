# Disaster Response Pipeline Project

### by Nham Nguyen Xuan
 
 
## Table of Contents

 1. [Summary](#summary)
 2. [File Description](#file-descriptions)
 3. [Usage](#usage)
 
## Summary

In this project, I analyzed disaster data using data engineering techniques. This information was used to create a model for an API that categorizes messages about disasters.

A data collection with actual communications that were sent during catastrophic occurrences is contained in the project folder. In order to categorize these occurrences and send the signals to the proper disaster aid organization, a machine learning pipeline was developed.

An emergency worker can enter a new message into this project's web interface and receive classification results in a number of categories. Additionally, data visualizations will be shown in the web application. This project demonstrates software abilities, such as writing simple data pipelines with clear, structured code.

## File Descriptions
app    

| - template    
| |- master.html # main page of web app    
| |- go.html # classification result page of web app    
|- run.py # Flask file that runs app    


data    

|- disaster_categories.csv # data to process    
|- disaster_messages.csv # data to process    
|- process_data.py # data cleaning pipeline    
|- InsertDatabaseName.db # database to save clean data to     


models   

|- train_classifier.py # machine learning pipeline     
|- classifier.pkl # saved model     


README.md 


## Usage

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

