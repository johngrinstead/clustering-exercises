import env

import pandas as pd

import sklearn

## 1. Acquire data from mall_customers.customers in mysql database

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
sql = '''
select *
from customers;
'''

## df = pd.read_sql(sql, get_connection('mall_customers')) ##

def acquire_mall():
    data = pd.read_csv("mall_customers.csv")
    
    return data


## 2. Split the data into train, validate, and split

def split(df, stratify_by=None):
    """
    Crude train, validate, test split
    To stratify, send in a column name
    """
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=.2, random_state=319)
        train, validate = train_test_split(train, test_size=.3, random_state=319)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=319, stratify=df[stratify_by])
        train, validate = train_test_split(train, test_size=.3, random_state=319, stratify=train[stratify_by])
    
    return train, validate, test



## 3. One-hot-encoding (pd.get_dummies)

def encode(df):
    dummy_df = pd.get_dummies(df[['gender']], dummy_na=False, drop_first=[True])
    
    dummy_df = dummy_df.rename(columns = {'gender_Male' : 'male'})   
    
    new_df = pd.concat([df, dummy_df], axis=1)
    
    new_df = new_df.drop(columns = 'gender')
    
    return new_df



## 4. Missing values

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


## 5. Scaling

def scale_data(df):
    scaler = sklearn.preprocessing.MinMaxScaler()
    
    scaler.fit(df)
    
    df_scaled = scaler.transform(df)
    
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    
    return df_scaled








