# IPL Data Analytics Project  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------
# Task 1: Load Dataset
# -------------------------------------------------------------
df = pd.read_csv("dataset/matches.csv", encoding="latin1")

# Drop useless column because it is fully empty
if "umpire3" in df.columns:
    df = df.drop(columns=["umpire3"])   

print("=== TASK 1: DATA UNDERSTANDING ===")
print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Identify quantitative and qualitative columns
quantitative_cols = df.select_dtypes(include=np.number).columns
qualitative_cols = df.select_dtypes(exclude=np.number).columns

print("\nQuantitative Columns:")
print(list(quantitative_cols))

print("\nQualitative Columns:")
print(list(qualitative_cols))

# ------------------------------------------------------------------
# Task 2: EDA
# ------------------------------------------------------------------
print("\n============ TASK 2: EDA ===============")

# 1. Top winning teams
plt.figure(figsize=(10, 5))
winner_counts = df["winner"].fillna("No Result").value_counts()
winner_counts.head(10).plot(kind="bar")
plt.title("Top 10 Winning Teams")
plt.xlabel("Team")
plt.ylabel("Wins")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Matches per season
plt.figure(figsize=(10, 5))
df["season"].value_counts().sort_index().plot(kind="bar")
plt.title("Matches Per Season")
plt.xlabel("Season")
plt.ylabel("Matches")
plt.tight_layout()
plt.show()

# 3. Histogram (Univariate)
plt.figure(figsize=(10, 5))
df["win_by_runs"].hist(bins=20)
plt.title("Histogram of Win By Runs")
plt.xlabel("Win By Runs")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Boxplot (Univariate / Outlier check)
plt.figure(figsize=(8, 5))
plt.boxplot(df["win_by_runs"].dropna())
plt.title("Boxplot of Win By Runs")
plt.ylabel("Win By Runs")
plt.tight_layout()
plt.show()

# 5. Scatter Plot (Bivariate)
plt.figure(figsize=(8, 5))
plt.scatter(df["win_by_wickets"], df["win_by_runs"])
plt.title("Win By Wickets vs Win By Runs")
plt.xlabel("Win By Wickets")
plt.ylabel("Win By Runs")
plt.tight_layout()
plt.show()

# 6. Correlation Matrix + Heatmap
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()

print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Task 3: Missing Values and Outliers
# ---------------------------------------------------------------
print("\n=============== TASK 3: MISSING VALUES HANDLING ===============")

df_clean = df.copy()

for col in df_clean.columns:
    if pd.api.types.is_numeric_dtype(df_clean[col]):
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    else:
        if df_clean[col].mode().empty:
            df_clean[col] = df_clean[col].fillna("Unknown")
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df_clean.isnull().sum())

# --------------------------------------------------------------------
# Task 4: Spread of Data
# --------------------------------------------------------------------
print("\n=== TASK 4: SPREAD OF DATA ===")

for col in numeric_df.columns:
    print(f"\nColumn: {col}")
    print("Mean:", df_clean[col].mean())
    print("Median:", df_clean[col].median())
    print("Standard Deviation:", df_clean[col].std())
    print("Skewness:", df_clean[col].skew())
    print("Kurtosis:", df_clean[col].kurt())

# --------------------------------------------------------------------
# Task 5: Automating EDA
# --------------------------------------------------------------------
print("\n=============== TASK 5: AUTOMATING EDA ===============")

def basic_eda(dataframe):
    print("\nDescribe:")
    print(dataframe.describe())

    print("\nInfo:")
    print(dataframe.info())

    print("\nNull Values:")
    print(dataframe.isnull().sum())

    print("\nCorrelation:")
    print(dataframe.select_dtypes(include=np.number).corr())

basic_eda(df_clean)

# --------------------------------------------------------------------
# Task 6: Simple Linear Regression
# --------------------------------------------------------------------
print("\n=============== TASK 6: SIMPLE LINEAR REGRESSION ===============")

simple_df = df_clean[["win_by_wickets", "win_by_runs"]]

X_simple = simple_df[["win_by_wickets"]]
y_simple = simple_df["win_by_runs"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

pred_simple = simple_model.predict(X_test_s)

print("Coefficient:", simple_model.coef_)
print("Intercept:", simple_model.intercept_)

# --------------------------------------------------------------------
# Task 7: Multiple Linear Regression
# --------------------------------------------------------------------
print("\n=== TASK 7: MULTIPLE LINEAR REGRESSION ===")

multi_df = df_clean[["win_by_wickets", "dl_applied", "win_by_runs"]]

X_multi = multi_df[["win_by_wickets", "dl_applied"]]
y_multi = multi_df["win_by_runs"]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)

pred_multi = multi_model.predict(X_test_m)

print("Coefficients:", multi_model.coef_)
print("Intercept:", multi_model.intercept_)

# --------------------------------------------------------------------
# Task 8 / 9: Overfitting and Underfitting
# --------------------------------------------------------------------
print("\n=== TASK 8 / 9: OVERFITTING AND UNDERFITTING ===")
print("Overfitting means model performs well on training data but poorly on testing data.")
print("Underfitting means model is too simple and performs poorly on both training and testing data.")

simple_train_score = simple_model.score(X_train_s, y_train_s)
simple_test_score = simple_model.score(X_test_s, y_test_s)

multi_train_score = multi_model.score(X_train_m, y_train_m)
multi_test_score = multi_model.score(X_test_m, y_test_m)

print("\nSimple Regression Train Score:", simple_train_score)
print("Simple Regression Test Score:", simple_test_score)

print("\nMultiple Regression Train Score:", multi_train_score)
print("Multiple Regression Test Score:", multi_test_score)

# ---------------------------------------------------------------------
# Task 10: Classification
# ---------------------------------------------------------------------
# print("\n=== TASK 10: CLASSIFICATION ===")

# class_df = df_clean[["team1", "team2", "winner"]].copy()

# le_team1 = LabelEncoder()
# le_team2 = LabelEncoder()
# le_winner = LabelEncoder()

# class_df["team1"] = le_team1.fit_transform(class_df["team1"])
# class_df["team2"] = le_team2.fit_transform(class_df["team2"])
# class_df["winner"] = le_winner.fit_transform(class_df["winner"])

# X_class = class_df[["team1", "team2"]]
# y_class = class_df["winner"]

# X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
#     X_class, y_class, test_size=0.2, random_state=42
# )

# class_model = LogisticRegression(max_iter=5000)
# class_model.fit(X_train_c, y_train_c)

# pred_class = class_model.predict(X_test_c)

# print("Accuracy:", accuracy_score(y_test_c, pred_class))
# print("Confusion Matrix:\n", confusion_matrix(y_test_c, pred_class))
# ---------------------------------------------------------------------
# Task 10: Logistic Regression Classification
# ---------------------------------------------------------------------
print("\n=== TASK 10: LOGISTIC REGRESSION CLASSIFICATION ===")

class_df = df_clean[["team1", "team2", "toss_winner", "toss_decision", "winner"]].copy()

# missing rows remove
class_df = class_df.dropna()

# label encoding
encoders = {}
for col in class_df.columns:
    le = LabelEncoder()
    class_df[col] = le.fit_transform(class_df[col].astype(str))
    encoders[col] = le

X_class = class_df[["team1", "team2", "toss_winner", "toss_decision"]]
y_class = class_df["winner"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train_c, y_train_c)

pred_class = log_model.predict(X_test_c)

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test_c, pred_class))
print("Confusion Matrix:\n", confusion_matrix(y_test_c, pred_class))

# ----------------------------------------------------------------------
# Task 11: Model Evaluation
# ----------------------------------------------------------------------
print("\n=============== TASK 11: MODEL EVALUATION ===============")

simple_mse = mean_squared_error(y_test_s, pred_simple)
simple_mae = mean_absolute_error(y_test_s, pred_simple)
simple_r2 = r2_score(y_test_s, pred_simple)

print("\nSimple Linear Regression:")
print("MSE:", simple_mse)
print("MAE:", simple_mae)
print("R2 Score:", simple_r2)

multi_mse = mean_squared_error(y_test_m, pred_multi)
multi_mae = mean_absolute_error(y_test_m, pred_multi)
multi_r2 = r2_score(y_test_m, pred_multi)

print("\nMultiple Linear Regression:")
print("MSE:", multi_mse)
print("MAE:", multi_mae)
print("R2 Score:", multi_r2)

# ------------------------------------------------------------------------
# Task 12: Final Model Performance
# ------------------------------------------------------------------------
print("\n=== TASK 12: FINAL PERFORMANCE INTERPRETATION ===")
print("Simple Regression R2:", simple_r2)
print("Multiple Regression R2:", multi_r2)
print("Classification Accuracy:", accuracy_score(y_test_c, pred_class))

# -----------------------------------------------------------------------
# Task 13: Final Visualization
# -----------------------------------------------------------------------
print("\n=============== TASK 13: FINAL VISUALIZATION ===============")

plt.figure(figsize=(10, 5))
df_clean["winner"].value_counts().head(15).plot(kind="bar")
plt.title("Top 15 Winners")
plt.xlabel("Team")
plt.ylabel("Wins")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n===================================== PROJECT COMPLETE =====================================")