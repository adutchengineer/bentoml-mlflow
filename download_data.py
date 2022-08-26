import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_data(csv:str) -> None:
    '''
    Separate and write the features to csv file
    '''
    df = pd.read_csv(csv)
    df = df.drop(['id', 'date'], axis=1)
    df = df.dropna()
    # split into input and output elements
    X = df.loc[:, df.columns != 'price']
    y = df.loc[:, 'price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)
   
    X_test.to_csv('data/test/X_test.csv',index=False)
    y_test.to_csv('data/test/y_test.csv',index=False)
    X_train.to_csv('data/train/X_train.csv',index=False)
    y_train.to_csv('data/train/y_train.csv',index=False) 

def main():
    data_path = 'data/kc_house_data.csv'
    if os.path.exists(data_path):
        split_data(data_path)
    else:
        print("Make sure to download the house sales data from kaggle(link: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) into data folder.")
        return


if __name__ == '__main__':
    main()