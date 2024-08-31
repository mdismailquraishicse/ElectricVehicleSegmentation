# import libraries
import pandas as pd
import eda


col = 'Selling_Price'
def binning(col):
    lower = df[col].quantile(.4)
    upper = df[col].quantile(.8)
    if x <= lower:
        return 'lowsp'
    elif x > lower and x < upper:
        return 'middlesp'
    else:
        return 'highsp'


def preprocessing(df):
    df['text'] = df['Car_Name'].apply(lambda x:x.replace(' ',''))