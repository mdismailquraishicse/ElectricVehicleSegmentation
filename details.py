# Import libraries
import numpy as np
import pandas as pd

path = 'data/car data.csv'
def getDf(path):
    df = pd.read_csv(path)
    print('First five rows :')
    print(df.head())
    print() # New line
    print('shape of the data :',df.shape)
    print('duplicate rows :', df.duplicated().sum())
    print()
    print('columns : ')
    print(list(df.columns))
    print()
    print('null values in each column :')
    null_df = df.isna().sum().reset_index(name='null_count')
    null_df['null_%'] = (df.isna().sum()/len(df)*100).values
    null_df.index = null_df['index']
    null_df.drop('index',axis=1,inplace=True)
    print(null_df)
    print()
    # Info
    print('INFO')
    print(df.info())
    print()
    # Describe
    print('Describe')
    print(df.describe())
    print()
    # Unique count
    print('# of unique values in each column :')
    print(df.nunique())
    print()
    print('*'*150)
    # Unique values in each column
    for col in df.columns:
        print(f'Unique values in column = {col} :')
        print(df[col].unique())
        print()
    return df

# getDf(path)