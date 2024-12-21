from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import pandas as pd
import dill

# Load your dataset (replace with your dataset path or dataframe)
data = pd.read_csv("heart.csv")  # Replace with your actual dataset path
X = data.drop("target", axis=1)  # Replace "target" with the actual target column name
y = data["target"]  # Replace "target" with the actual target column name

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the ClusterSimilarity class
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# Define custom feature transformations
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

ratio_pipeline = lambda: make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(column_ratio, feature_names_out=ratio_name),
    StandardScaler()
)

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
    StandardScaler()
)

# Define preprocessing pipeline
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

# Dynamically select columns for log transformation
log_columns = [col for col in ["age", "cholesterol", "trestbps"] if col in X_train.columns]

# Define preprocessing pipeline
preprocessing = make_column_transformer(
    (default_num_pipeline, make_column_selector(dtype_include=np.number)),  # Numerical columns
    (OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include=object)),  # Categorical columns
    (log_pipeline, log_columns),  # Log transformation for valid columns
    remainder="passthrough"  # Keep remaining columns as they are
)


# Define model pipelines
pipelines = {
    "ridge": make_pipeline(preprocessing, Ridge()),
    "rf": make_pipeline(preprocessing, RandomForestRegressor(random_state=42)),
    "gb": make_pipeline(preprocessing, GradientBoostingRegressor(random_state=42)),
}

# Define hyperparameter grids
param_grids = {
    "ridge": {"ridge__alpha": [0.05, 0.25, 0.5, 1.0]},
    "rf": {
        "randomforestregressor__n_estimators": [50, 100, 150],
        "randomforestregressor__max_depth": [5, 10, 15],
    },
    "gb": {
        "gradientboostingregressor__n_estimators": [50, 100, 150],
        "gradientboostingregressor__learning_rate": [0.01, 0.1, 0.2],
    },
}

# Perform GridSearchCV for each pipeline
fit_models = {}
results = []
for algo, pipeline in pipelines.items():
    grid_search = GridSearchCV(
        pipeline, param_grids[algo], cv=3, n_jobs=-1, scoring="r2", verbose=1
    )
    grid_search.fit(X_train, y_train)
    fit_models[algo] = grid_search
    results.append({
        "Model": algo,
        "Best Parameters": grid_search.best_params_,
        "Best CV Score (R2)": grid_search.best_score_,
    })

# Summary
results_df = pd.DataFrame(results)
print(results_df)

# Save the best model
best_model_name = max(results, key=lambda x: x["Best CV Score (R2)"])["Model"]
best_model = fit_models[best_model_name].best_estimator_

# Save using dill
with open("best_model_heart_prediction.pkl", "wb") as f:
    dill.dump(best_model, f)