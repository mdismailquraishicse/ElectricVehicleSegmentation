# Import libraries
import numpy as np
import pandas as pd

path = 'data/car data.csv'

def getMeanMedianMax(df,by_, value_):
    df_desc = df[df[by_]==value_].describe()
    result = pd.concat([df_desc.iloc[1],df_desc.iloc[5],df_desc.iloc[7]],axis=1)
    result.drop('Owner', inplace=True)
    result.rename({'50%':'median'},axis=1,inplace=True)
    return result
    
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
    # Unique values in each column
    for col in df.columns:
        print(f'Unique values in column = {col} :')
        print(df[col].unique())
        print()
    return df

def EDA(df):
    # Top 5 values with highest count
    df.drop_duplicates(inplace=True)
    print('Duplicate rows successfully dropped')
    categorical_columns = list(df.describe(include='object').columns)
    categorical_columns.append('Owner')
    # Mean, Median, Max
    print('Top 5 car names by highest selling price')
    print(df.groupby(['Car_Name'])['Selling_Price'].mean().sort_values(ascending=False).head())
    print()
    print('Top 5 car names by highest present price')
    print(df.groupby(['Car_Name'])['Present_Price'].mean().sort_values(ascending=False).head())
    print('-'*150)
    print('Mean Median Max of Fuel Type = Petrol')
    print(getMeanMedianMax(df,'Fuel_Type','Petrol'))
    print()
    print('Mean Median Max of Fuel Type = Diesel')
    print(getMeanMedianMax(df,'Fuel_Type','Diesel'))
    print()
    print('Mean Median Max of Fuel Type = CNG')
    print(getMeanMedianMax(df,'Fuel_Type','CNG'))
    print('-'*150)
    for col in categorical_columns:
        print(f'Number of cars for each {col}:')
        print(df[col].value_counts().reset_index().head())
        print() # prints new line
    print('-'*150)
    # Mean selling price, present price, kms_driven for each year
    print('Average Selling price, Present Price and KMs Driven in each year :')
    print(round(df.groupby(['Year'])[['Selling_Price', 'Present_Price', 'Kms_Driven']].mean(),2))
    print('-'*150)
    print('Mean Selling Price for each Fuel Type, Seller Type and Transmission')
    print(round(df.groupby(['Fuel_Type','Seller_Type', 'Transmission'])[['Selling_Price']].mean(),2).unstack())