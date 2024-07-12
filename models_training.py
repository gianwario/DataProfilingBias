import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler


def eq_odds_fair_report(dataset, prediction):
    dataset["income"] = y
    results = {}
    sex_features = ['sex_Male','sex_Female']

    aif_sex_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['income'],
        protected_attribute_names=sex_features,
        privileged_protected_attributes=['sex_Male'],
    )

    aif_sex_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['income'],
        protected_attribute_names=sex_features,
        privileged_protected_attributes=['sex_Male'],
    )

    sex_privileged_groups = [{'sex_Male': 1}]
    sex_unprivileged_groups = [{'sex_Female': 1}]

    metrics = ClassificationMetric(dataset=aif_sex_dataset,classified_dataset=aif_sex_pred,unprivileged_groups=sex_unprivileged_groups,privileged_groups=sex_privileged_groups)

    results['sex_mean_diff'] = round(metrics.mean_difference(),3)
    results['sex_eq_opp_diff'] = round(metrics.equal_opportunity_difference(),3)
    results['sex_avg_odds_diff'] = round(metrics.average_odds_difference(),3)

    race_feature = ['race']

    aif_race_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['income'],
        protected_attribute_names=race_feature,
    )

    aif_race_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['income'],
        protected_attribute_names=race_feature,
    )

    race_privileged_groups = [{'race': 1}]
    race_unprivileged_groups = [{'race': 0}]

    metrics = ClassificationMetric(dataset=aif_race_dataset,classified_dataset=aif_race_pred,unprivileged_groups=race_unprivileged_groups,privileged_groups=race_privileged_groups)

    results['race_mean_diff'] = round(metrics.mean_difference(),3)
    results['race_eq_opp_diff'] = round(metrics.equal_opportunity_difference(),3)
    results['race_avg_odds_diff'] = round(metrics.average_odds_difference(),3)

    protected_features = ['sex_Male','sex_Female','race']

    aif_overall_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['income'],
        protected_attribute_names=protected_features,
    )

    aif_overall_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['income'],
        protected_attribute_names=protected_features,
    )

    privileged_groups = [{'sex_Male': 1} | {'race': 1}]
    unprivileged_groups = [{'sex_Female': 1, 'race': 0}]

    metrics = ClassificationMetric(dataset=aif_overall_dataset,classified_dataset=aif_overall_pred,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
    results['overall_mean_diff'] = round(metrics.mean_difference(),3)
    results['overall_eq_opp_diff'] = round(metrics.equal_opportunity_difference(),3)
    results['overall_avg_odds_diff'] = round(metrics.average_odds_difference(),3)

    print(results)


if __name__ == "__main__":
    df = pd.read_csv("output_datasets/adult_condition3.csv")

    # cancelliamo dal dataset le entry con attributi mancanti
    df = df.dropna()

    # lista di feature numeriche che non necessitano di modifiche
    numeric_features = [
        'fnlwgt','capital-gain','capital-loss'
    ]
    # lista delle feature categoriche
    categorical_features = [
        'workclass', 'education','hours-range', 'occupation', 'relationship','age-range','sex', 'nc'
    ]

    # lista nomi delle feature del dataset
    features = df.columns.to_list()

    # ciclo per rimuovere le feature numeriche dalla lista delle feature
    for num_feature in numeric_features:
        features.remove(num_feature)


    # rimpiazziamo i valori categorici della nostra variabile target con valori numerici
    # 1 per salario maggiore di 50K, 0 altrimenti
    df['income'] = df['income'].replace('<=50K',0)
    df['income'] = df['income'].replace('>50K',1)
    df['race'] = df['race'].replace({'White':1,'Black':0,'Asian-Pac-Islander':0,'Amer-Indian-Eskimo':0,'Other':0})

    # encoding delle feature categoriche in valori numerici tramite la funzione get_dummies()
    # funzione che crea X nuove colonne, con X numero di possibili valori per la feature, impostando valore 0.0 e 1.0
    one_hot = pd.get_dummies(df[categorical_features], dtype=int)

    df.drop('id',inplace=True,axis=1)
    # Cancelliamo dal DataFrame originale le features categoriche espanse per poi unire alle rimanenti il nuovo dataframe ottenuto dalla funzione
    # get_dummies()
    df = df.drop(categorical_features, axis=1)
    df = df.join(one_hot)

    
    dataset = df
    # setting nomi features del dataset
    features = dataset.columns.tolist()

    # rimuoviamo il nome della feature target dalla lista nomi features
    features.remove('income')

    # setting nome target feature
    target = ['income']

    fair_results = {}

    # setting dataset features
    X = dataset[features]
    # setting dataset target feature
    y = dataset[target]


    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train)

    # Initialize models
    models = {
            'Logistic Regression': make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),
            'Random Forest': make_pipeline(StandardScaler(),RandomForestClassifier()),
            'Linear SVC': make_pipeline(StandardScaler(),LinearSVC(dual=True,max_iter=1000)),
            'XGBoost': make_pipeline(StandardScaler(), XGBClassifier(objective='binary:logistic', random_state=42)),
    }

    # Train and evaluate models
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train.values, y_train.values.ravel())
        y_pred = model.predict(X_test)


        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"\n{model_name} Accuracy: {accuracy:.4f}")
        print(f"Classification Report for {model_name}:")
        print(report)
        
        X_test_df = X_test.copy(deep=True)
        X_test_df['income'] = y_test

        pred = X_test_df.copy(deep=True)
        pred['income'] = y_pred

        print("Fairness Results")

        eq_odds_fair_report(X_test_df, pred)
        print("=" * 50)
