import prepro
import pandas as pd
from sklearn.cluster import KMeans
import eda
import prepro
import pickle
# Path
path = 'data/car data.csv'
df = pd.read_csv(path) # Load df
# Function for model building and save as pickle
def getModel(df):
    df.drop_duplicates(inplace=True)
    df, vector_df, countvectorizer = prepro.preBuild(df)
    model = KMeans(n_clusters=8, random_state=1234)
    model.fit(vector_df.values)
    print('Model successfully trained')
    pickle.dump(model,open('models/vehicle_segmentation_model.pkl','wb'))
    pickle.dump(countvectorizer,open('models/countvectorizer.pkl','wb'))
    print('model saved as models/vehicle_segmentation_model.pkl')
# Load pickel models and make prediction
def makePred(data):
    vectorizer = pickle.load(open('models/countvectorizer.pkl','rb'))
    model = pickle.load(open('models/vehicle_segmentation_model.pkl','rb'))
    print('model successfully loaded')
    transformed_data = vectorizer.transform(data)
    prediction = model.predict(transformed_data)
    return prediction

if  __name__ == '__main__':
    getModel(df)