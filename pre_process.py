import pandas as pd


'''
The majority of the missing values belong to attributes that are not relevant for our analysis (e.g., “Marital-Status”), and therefore decided to
perform feature selection first and then remove the few tuples that still contain missing values.
'''
def remove_features(dataset: pd.DataFrame):
    dataset = dataset.drop(columns=["marital-status", "education-num"])
    return dataset



'''
"To extract Functional Dependencies that do not depend on specific values of a continuous
 or rational attribute, it is often useful to group the values into appropriately defined bins"
 Following this statement, we perform Discretization
'''
def discretize(dataset: pd.DataFrame):
    hpw = pd.cut(dataset["hours-per-week"], bins=[20,40,60,80,100])
    age = pd.cut(dataset["age"], bins=[15,30,45,60,75,100])
    dataset = dataset.drop(columns=["hours-per-week", "age"])
    dataset["age-range"] = age
    dataset["hours-range"] = hpw
    return dataset


'''
We remove NAs and duplicates
'''
def clean_dataset(dataset: pd.DataFrame):
    dataset = dataset.dropna()
    dataset = dataset.drop_duplicates()
    return dataset

'''
To have readable and more effective Functional Dependencies, we keep the attribute “Race” as is in the original dataset and
group the values of the attribute “NC” using four different values: “NC-US,” “NC-Hispanic,” “NCNon-US-Hispanic,” and “NC-Asian-Pacific.”
'''
def process_native_country(dataset: pd.DataFrame):
    nc = []
    ncus = ['United-States', 'Jamaica', 'South', 'Canada', 'Outlying-US(Guam-USVI-etc)']
    nchisp = ['Mexico', 'Puerto-Rico', 'Columbia', 'Ecuador', 'Dominican-Republic', 'Peru']
    ncasia = ['India', 'Honduras', 'Philippines', 'Cambodia', 'Thailand', 'Laos', 'Taiwan', 'Haiti', 'El-Salvador',  'Guatemala', 'Nicaragua', 'Vietnam', 'Hong']

    for element in dataset["native-country"]:
        if element in ncus:
            nc.append("NC-US")
        elif element in nchisp:
            nc.append("NC-Hispanic")
        elif element in ncasia:
            nc.append("NC-Asian-Pacific")
        else:
            nc.append("NCNon-US-Hispanic")

    dataset = dataset.drop(columns=["native-country"])
    dataset["nc"] = nc
    return dataset