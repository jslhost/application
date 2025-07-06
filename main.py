import os
import argparse
from dotenv import load_dotenv
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix

# Définitions des variables
load_dotenv()
jeton_api = os.environ.get("JETON_API", "")
data_path = os.environ.get("DATA_PATH", "")

parser = argparse.ArgumentParser(description="Nombre d'arbres de la RandomForest")
parser.add_argument("--trees", type=int, default=20)
args = parser.parse_args()
n_trees = args.trees
print(f"Nb trees: {n_trees}")

numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]
max_depth = None
max_features = "sqrt"


def preprocess_and_model_pipe(
    n_trees=n_trees,
    numeric_features=None,
    categorical_features=None,
    max_depth=max_depth,
    max_features=max_features,
):

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("Preprocessing numerical", numeric_transformer, numeric_features),
            (
                "Preprocessing categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_trees, max_depth=max_depth, max_features=max_features
                ),
            ),
        ]
    )

    return pipe


def model_evaluation(model, X_train, y_train, X_test, y_test):
    model_score_test = model.score(X_test, y_test)
    model_score_train = model.score(X_train, y_train)
    conf_matrix = confusion_matrix(y_test, model.predict(X_test))

    return model_score_train, model_score_test, conf_matrix


if __name__ == "__main__":

    data = pd.read_csv(data_path)
    y = data["Survived"]
    X = data.drop("Survived", axis="columns")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model_pipeline = preprocess_and_model_pipe(
        n_trees=n_trees,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        max_depth=max_depth,
        max_features=max_features,
    )

    # Preprocessing, training et évaluation
    model_pipeline.fit(X_train, y_train)

    model_score_train, model_score_test, conf_matrix = model_evaluation(
        model_pipeline, X_train, y_train, X_test, y_test
    )

    # Affichage des résultats
    print(f"{model_score_train:.1%} de bonnes réponses sur les données de train")
    print(f"{model_score_test:.1%} de bonnes réponses sur les données de test")

    print(20 * "-")
    print("matrice de confusion")
    print(conf_matrix)
