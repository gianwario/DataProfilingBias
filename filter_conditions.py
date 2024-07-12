import pandas as pd

'''
This filters creates a new dataset in which holds this conditions:
People that work in Private and have at least a Bechelor level of education

6555 rows
'''
def filter1(dataset: pd.DataFrame):
    dataset = dataset.loc[(dataset['workclass'].eq("Private")) 
        &
        (dataset['education'].eq("Bachelors") | dataset['education'].eq("Masters") | dataset['education'].eq("Doctorate") )]
    dataset = dataset.drop(columns=["Unnamed: 0"])
    dataset.to_csv("output_datasets/adult_condition1.csv")


'''
This filters creates a new dataset in which holds this conditions:
People that are from North America and are not married

14739 rows
'''
def filter2(dataset: pd.DataFrame):
    dataset = dataset.loc[(dataset['nc'].eq("NC-US")) 
        &
        (dataset['relationship'].eq("Not-in-family") | dataset['relationship'].eq("Unmarried") )]
    dataset = dataset.drop(columns=["Unnamed: 0"])
    dataset.to_csv("output_datasets/adult_condition2.csv")


'''
This filters creates a new dataset in which holds this conditions:
People that are married, have no Bachelor/Masters/Doctoral degree and come from non American countries (europe, hispanic or asia)

1879 rows
'''
def filter3(dataset: pd.DataFrame):
    dataset = dataset.loc[(dataset['relationship'].eq("Wife") | dataset['relationship'].eq("Husband")) 
        &
       (dataset['education'].ne("Bachelors") | dataset['education'].ne("Masters") | dataset['education'].ne("Doctorate") )
        &
        (dataset['nc'].ne("NC-US")) ]
    dataset = dataset.drop(columns=["Unnamed: 0"])
    dataset.to_csv("output_datasets/adult_condition3.csv")