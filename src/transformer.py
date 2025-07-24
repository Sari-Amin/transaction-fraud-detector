from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd

class DataTransformer:
    def __init__(self, df, target_col):
        self.df = df.copy()
        self.target_col = target_col
        self.X = self.df.drop(columns=[target_col])
        self.y = self.df[target_col]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pipeline = None
        self.feature_names = None
    def split_data(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, stratify=self.y, random_state=42
        )


    def encode_and_scale(self):
        numeric_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Identify and remove high-cardinality categorical features
        high_cardinality = [col for col in categorical_cols if self.X[col].nunique() > 100]
        categorical_cols = [col for col in categorical_cols if col not in high_cardinality]

        if high_cardinality:
            print("Dropping high-cardinality columns:", high_cardinality)

        # Drop them from X
        self.X.drop(columns=high_cardinality, inplace=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        self.pipeline = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

        self.X_train = self.pipeline.fit_transform(self.X_train)
        self.X_test = self.pipeline.transform(self.X_test)

        # Save feature names
        cat_features = self.pipeline.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        self.feature_names = numeric_cols + cat_features.tolist()




    def handle_class_imbalance(self, method="smote"):
        if method == "smote":
            sampler = SMOTE(random_state=42)
        elif method == "undersample":
            sampler = RandomUnderSampler(random_state=42)
        else:
            raise ValueError("Invalid method")

        self.X_train, self.y_train = sampler.fit_resample(self.X_train, self.y_train)

    def get_transformed_data(self):
        self.split_data()
        self.encode_and_scale()
        self.handle_class_imbalance()
        return self.X_train, self.X_test, self.y_train, self.y_test


# df = pd.read_csv('data/Fraud_Data.csv', parse_dates=['purchase_time', "signup_time"])
# ip = pd.read_csv('data/IpAddress_to_Country.csv')

# print("Testing")
# f = DataTransformer(df,'class')
# X_train, X_test, y_train, y_test = f.get_transformed_data()
# print(X_train.shape, y_train.value_counts())
