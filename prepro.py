# import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import eda
# Binning Selling Price
def binSp(x):
    lower = 0.85
    upper = 6.0
    if x <= lower:
        return 'lowsp'
    elif x > lower and x < upper:
        return 'midsp'
    else:
        return 'highsp'
# Binning Kms_Driven
def binKms(x):
    lower = 18000
    upper = 53000
    if x <= lower:
        return 'lowkms'
    elif x > lower and x < upper:
        return 'midkms'
    else:
        return 'highkms'
# Data preprocessing for model building
def preBuild(df):
    df['SellingPriceRange'] = df['Selling_Price'].apply(binSp)
    df['KmsDrivenRange'] = df['Kms_Driven'].apply(binKms)
    df['text'] = df['Car_Name'].apply(lambda x:x.replace(' ',''))
    # df['text'] = df['text']+' '+df['Year'].astype('str')+' '+df['SellingPriceRange']+' '+df['KmsDrivenRange']+\
    # ' '+df['Fuel_Type']+' '+df['Seller_Type']+' '+df['Transmission']+' '+df['Owner'].astype('str')
    df['text'] = df['text']+' '+df['Year'].astype('str')+' '+df['SellingPriceRange']+' '+df['KmsDrivenRange']+\
    ' '+df['Fuel_Type']+' '+df['Transmission']
    df['text'] = df['text'].apply(lambda x:x.lower())
    countvectorizer = CountVectorizer(max_features=50, min_df=3)
    text_vectors = countvectorizer.fit_transform(df['text']).toarray()
    print('total features : ',len(text_vectors[0]))
    vec_df = pd.DataFrame(data =text_vectors , columns = countvectorizer.get_feature_names_out())
    print('-*-'*20,end=' ')
    print('END OF PREPROCESSING',end=' ')
    print('-*-'*20)
    return df, vec_df, countvectorizer