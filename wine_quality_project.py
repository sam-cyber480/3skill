import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def print_separator(title):
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50 + "\n")

# ==========================================
# TASK 1: Load and Understand the Dataset
# ==========================================
print_separator("TASK 1: Load and Understand the Dataset")

# Load dataset (Note: sep=';' is required for this dataset)
try:
    df = pd.read_csv('winequality.csv', sep='\t')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: winequality.csv not found. Please ensure the file is in the same directory.")
    exit()

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Last 5 rows ---")
print(df.tail())

print("\n--- Random 5 rows ---")
print(df.sample(5, random_state=42))

print("\n[Explanation]")
print("The dataset contains chemical properties of red wine variants.")
print("Each row represents a specific wine sample with its physicochemical tests (inputs) and sensory quality rating (output).")


# ==========================================
# TASK 2: Basic Data Inspection
# ==========================================
print_separator("TASK 2: Basic Data Inspection")

print(f"Column Names: {df.columns.tolist()}")
print(f"Shape: {df.shape} (Rows, Columns)")
print("\n--- Data Types ---")
print(df.dtypes)
print("\n--- Summary Statistics ---")
print(df.describe())

print("\n[Explanation]")
print("Data inspection is crucial to identify data types, ranges, scales, and potential anomalies.")
print("It helps in deciding preprocessing steps like scaling, encoding, or outlier removal.")


# ==========================================
# TASK 3: Missing Values Analysis
# ==========================================
print_separator("TASK 3: Missing Values Analysis")

missing_values = df.isnull().sum()
print("\n--- Missing Values Count ---")
print(missing_values)

has_missing = missing_values.sum() > 0
print(f"\nAre there missing values? {'Yes' if has_missing else 'No'}")

print("\n[Explanation]")
if not has_missing:
    print("Fortunately, there are no missing values.")
else:
    print("There are missing values.")
print("In a real-world project, missing values could be handled by:")
print("1. Imputation (Mean/Median/Mode) for numerical/categorical data.")
print("2. Dropping rows/columns if missingness is high.")
print("3. Using advanced techniques like KNN imputation.")


# ==========================================
# TASK 4: Exploratory Data Analysis (EDA)
# ==========================================
print_separator("TASK 4: Exploratory Data Analysis (EDA)")

print("\n--- Quality Value Counts ---")
print(df['quality'].value_counts().sort_index())

plt.figure(figsize=(8, 6))
sns.countplot(x='quality', data=df)
plt.title('Distribution of Wine Quality')
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.savefig('quality_distribution_plot.png')
print("\n[Plot saved as 'quality_distribution_plot.png']")

print("\n[Observations]")
print("1. The dataset is imbalanced. Most wines have average quality (5 or 6).")
print("2. Very few wines have extremely high (8) or low (3) quality scores.")
print("3. It might be harder for a model to predict the minority classes (3, 4, 7, 8) accurately.")

print("\n[Explanation]")
print("EDA is vital to understand the distribution of the target variable and relationships between features.")
print("It helps identifying class imbalance, which informs metric selection and sampling strategies.")


# ==========================================
# TASK 5: Convert the Problem into a Classification Task
# ==========================================
print_separator("TASK 5: Convert to Binary Classification")

# Quality >= 7 -> Good (1), else Bad (0)
df['quality_label'] = (df['quality'] >= 7).astype(int)

print("Created 'quality_label' column.")
print("\n--- New Class Distribution ---")
print(df['quality_label'].value_counts())

print("\n[Explanation]")
print("Binary classification simplifies the problem into 'Good' vs 'Not Good', which is often more actionable for consumers.")
print("Predicting exact scores is harder and sometimes less necessary than knowing if a wine is simply 'premium' or not.")


# ==========================================
# TASK 6: Feature and Target Separation
# ==========================================
print_separator("TASK 6: Feature and Target Separation")

X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

print("\n[Explanation]")
print("We exclude 'quality' because it is the target source itself (leakage).")
print("We exclude 'quality_label' from features because it IS the target we want to predict.")


# ==========================================
# TASK 7: Train–Test Split
# ==========================================
print_separator("TASK 7: Train–Test Split")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

print("\n[Explanation]")
print("Splitting data allows us to evaluate the model on unseen data.")
print("Testing on training data causes overfitting, where the model memorizes data instead of generalizing.")


# ==========================================
# TASK 8: Feature Scaling
# ==========================================
print_separator("TASK 8: Feature Scaling")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler.")

print("\n[Explanation]")
print("Scaling ensures all features contribute equally to distance calculations.")
print("Models relying on distances (KNN, SVM) or gradients (Logistic Regression, Neural Nets) require scaling.")
print("Tree-based models (Decision Trees, Random Forest) are generally invariant to scaling.")


# ==========================================
# TASK 9: Model Training
# ==========================================
print_separator("TASK 9: Model Training")

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC()
}

results = {}

print("Training models...")

for name, model in models.items():
    # Use scaled data for non-tree models, though trees handle scaled data fine too.
    # To be precise: LogReg, KNN, SVM need scaled. Trees don't strictly need it but it doesn't hurt.
    # We will use scaled data for all for consistency here, as it's safe.
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"\n{name}:")
    if name == "Logistic Regression":
        print(" - Linear model that estimates probability using a logistic function.")
    elif name == "KNN":
        print(" - Classifies based on the majority class of closest neighbors.")
    elif name == "Decision Tree":
        print(" - Splits data based on feature values to create a tree structure.")
    elif name == "Random Forest":
        print(" - Ensemble of many decision trees to reduce overfitting.")
    elif name == "SVM":
        print(" - Finds the best hyperplane to separate classes.")


# ==========================================
# TASK 10: Model Evaluation and Comparison
# ==========================================
print_separator("TASK 10: Model Evaluation and Comparison")

results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
results_df = results_df.sort_values(by='Accuracy', ascending=False)

print(results_df)

best_model_name = results_df.iloc[0]['Model']
print(f"\nBest performing model: {best_model_name}")
print("[Reasoning]")
print("Random Forest typically performs well on tabular data as it handles non-linearities and reduces variance.")
print("SVM also tends to work well with high dimensional spaces, but Random Forest is often more robust out-of-the-box.")


# ==========================================
# TASK 11: Pipeline and Hyperparameter Tuning
# ==========================================
print_separator("TASK 11: Pipeline and Hyperparameter Tuning")

# Using SVM for tuning example
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf']
}

print(f"Tuning SVM with GridSearchCV...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train) # Pipeline handles scaling internally

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

print("\n[Explanation]")
print("Pipelines prevent data leakage by ensuring preprocessing (like scaling) happens within each fold of CV.")
print("Hyperparameter tuning finds the optimal configuration to maximize model performance.")


# ==========================================
# TASK 12: Final Conclusion
# ==========================================
print_separator("TASK 12: Final Conclusion")

print("1. Dataset: The Red Wine Quality dataset contains chemical attributes and quality ratings.")
print("2. Observations: Features like alcohol and volatile acidity often correlate with quality.")
print(f"3. Best Model: {best_model_name} achieved the highest accuracy in initial testing.")
print("4. Learning: This project demonstrated the end-to-end ML workflow from cleaning to tuning.")
print("5. Real-world: Similar steps are used in industry for credit scoring, churn prediction, etc.")

print("\nProject execution complete.")