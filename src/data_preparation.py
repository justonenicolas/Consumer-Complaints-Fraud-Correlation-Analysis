"""
Data exploration and preparation functions
"""

import pandas as pd
import unicodedata2 as unicodedata 
import re
import os
from kaggle.api.kaggle_api_extended import KaggleApi
from IPython.display import display
import sys
sys.path.insert(0, "..")
from src.config import (RAW_DATA_DIR, ROOT_DIR, PROCESSED_DATA_DIR)

# Show all columns 
pd.set_option('display.max_columns', None)

# Bold Text Setup
b_start = '\033[1m'
b_end = '\033[0m'

# Regular Expressions Compilation
ANONYMOUS_CLEAN = re.compile(r'[xX]{2,}')
SPECIAL_CHARS = re.compile(r'[^A-Z0-9\s.]')
DOTS = re.compile(r'\.+\s*')
FINAL_LETTER = re.compile(r'\s+[A-Z]$')
MULTI_SPACE = re.compile(r'\s+')
HTML_CLEAN = re.compile(r'<.*?>')
URL_CLEAN = re.compile(r'http\S+')
CLEAN_NUMBERS = re.compile(r'\d+')

# Download datset from Kaggle
def load_data():
    data_path = fr'{RAW_DATA_DIR}\consumer_complaints.csv'
    
    if not os.path.exists(data_path):
        print("Downloading dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate() # Requiere que kaggle.json esté en ~/.kaggle/
        
        api.dataset_download_files(
            "justonenicolas/consumer-complaints-2011-2022", 
            path=RAW_DATA_DIR, 
            unzip=True
        )
        print(f"Succesfully downloaded to {RAW_DATA_DIR}")
    else:
        print(f"Dataset already exist in {RAW_DATA_DIR}")

if __name__ == "__main__":
    load_or_download_data()


# Data Exploration function
def explore_data(
    df,
    explore_features=None,
    unique_features=None
):
    """
    Quick exploratory analysis of a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to explore.
    explore_features : list[str], optional
        Columns for duplicate analysis.
    unique_features : list[str], optional
        Columns whose unique values should be displayed.
    """

    explore_features = explore_features or []
    unique_features = unique_features or []

    # Quick data exploration   
    print(f'{b_start}=== DataFrame General Info ==={b_end}')
    display(df.info())
    print()
    print(f'{b_start}=== DataFrame Sample ==={b_end}')
    display(df.sample(3))
    print()

    # Missing values exploration
    null_count = pd.DataFrame(df.isna().sum().sort_values(ascending=False), columns=['null_count'])
    null_count['%_null_count'] = ((null_count['null_count']/len(df))*100).round(2)
    
    print(f'{b_start}=== Fields with missing values ==={b_end}')
    display(null_count.query('null_count > 0'))
    print()

    # Duplicate values exploration
    print(f'{b_start}=== Duplicate value counts ==={b_end}')
    display(df.duplicated().value_counts())
    print()

    if len(explore_features) > 0:
        for feature in explore_features:
            print(f'{b_start}=== Duplicate value counts for {feature} ==={b_end}')
            display(df[feature].duplicated().value_counts())
            print()

    # Unique values exploration
    if len(unique_features) > 0:
        for feature in unique_features:
            print(f'{b_start}=== Unique values for {feature} ==={b_end}')
            display(df[feature].sort_values().unique())
            print()


# Text Cleaning function
def normalize_text(name):
    """
    Company name cleaning and upper casing

    Parameters
    ----------
    name : str or any
        Value to normalize. Non-string values will be converted to string.
    """
    if pd.isna(name):
       return ""

    name = str(name).upper().strip()

    # Remove special characteres
    name = ''.join(
        c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn'
    )
    
    # Specific replacements
    name = name.replace('&', ' AND ')
    name = ANONYMOUS_CLEAN.sub('', name)
    name = SPECIAL_CHARS.sub(' ', name)
    name = DOTS.sub('', name)
    name = FINAL_LETTER.sub('', name)
    name = HTML_CLEAN.sub('', name)
    name = URL_CLEAN.sub('', name)
    name = CLEAN_NUMBERS.sub('', name)
    name = MULTI_SPACE.sub(' ', name).strip()
    return name