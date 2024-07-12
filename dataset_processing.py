from ucimlrepo import fetch_ucirepo 
import pandas as pd
from pre_process import discretize, clean_dataset, remove_features, process_native_country
from filter_conditions import filter1, filter2, filter3


'''
This method fetches the UCI ML repository to download the Adult dataset and saves it locally
'''
def save_dataset():
    # fetch dataset 
    adult = fetch_ucirepo(id=2) 
  
    # data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets 
    
    # metadata 
    print(adult.metadata) 
    
    # variable information 
    print(adult.variables) 

    dataset = pd.DataFrame(adult.data.original, columns=adult.data.headers)
    dataset.to_csv("adult.csv")


def data_visualization(dataset):
    print(dataset["workclass"].unique())
    print(dataset["relationship"].unique())
    print(dataset["education"].unique())
    print(dataset["nc"].unique())

if __name__ == "__main__":
    #save_dataset()
    dataset = pd.read_csv("adult.csv")
    # 48841 initial rows
    dataset = remove_features(dataset)
    dataset = discretize(dataset)
    dataset = clean_dataset(dataset)
    dataset = process_native_country(dataset)
    print(dataset)
    # 43479 rows remaining after data preparation

    # We visualize some feature to understand the possible conditions to apply
    data_visualization(dataset)

    # We create 3 filtered datasets to apply conditions
    filter1(dataset)
    filter2(dataset)
    filter3(dataset)
    pass