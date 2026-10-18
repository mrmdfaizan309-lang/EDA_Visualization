# ============================================================
# WEEK 2 TASK
# Exploratory Data Analysis and Visualization
# Dataset: Titanic
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# 1. CREATE VISUALIZATION FOLDER
# ------------------------------------------------------------

os.makedirs("visualizations", exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

file_path = "data/titanic_cleaned.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET LOADED SUCCESSFULLY ==========")
print(df.head())


# ------------------------------------------------------------
# 3. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== DATASET INFORMATION ==========")
df.info()


# ------------------------------------------------------------
# 4. STATISTICAL SUMMARY
# ------------------------------------------------------------

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


# ------------------------------------------------------------
# 5. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ------------------------------------------------------------
# 6. SURVIVAL COUNT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Survived")

plt.title("Titanic Passenger Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig("visualizations/survival_count.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 7. SURVIVAL BY GENDER
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Sex", hue="Survived")

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])

plt.tight_layout()
plt.savefig("visualizations/survival_gender.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 8. SURVIVAL BY PASSENGER CLASS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Pclass", hue="Survived")

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])

plt.tight_layout()
plt.savefig("visualizations/survival_class.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 9. AGE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="Age", kde=True)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("visualizations/age_distribution.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 10. FARE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="Fare", kde=True)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("visualizations/fare_distribution.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 11. AGE VS FARE SCATTER PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Age",
    y="Fare",
    hue="Survived"
)

plt.title("Relationship Between Age and Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(title="Survived")

plt.tight_layout()
plt.savefig("visualizations/age_fare_scatter.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 12. CORRELATION HEATMAP
# ------------------------------------------------------------

numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("visualizations/correlation_heatmap.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 13. AGE DISTRIBUTION BY PASSENGER CLASS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Pclass",
    y="Age"
)

plt.title("Age Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Age")

plt.tight_layout()
plt.savefig("visualizations/age_class_boxplot.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 14. AVERAGE FARE BY PASSENGER CLASS
# ------------------------------------------------------------

average_fare = df.groupby("Pclass")["Fare"].mean()

print("\n========== AVERAGE FARE BY CLASS ==========")
print(average_fare)

plt.figure(figsize=(8, 5))

average_fare.plot(kind="bar")

plt.title("Average Fare by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Average Fare")

plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("visualizations/average_fare_class.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 15. SURVIVAL RATE BY GENDER
# ------------------------------------------------------------

survival_gender = df.groupby("Sex")["Survived"].mean() * 100

print("\n========== SURVIVAL RATE BY GENDER ==========")
print(survival_gender)

plt.figure(figsize=(8, 5))

survival_gender.plot(kind="bar")

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")

plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("visualizations/survival_rate_gender.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 16. SURVIVAL RATE BY PASSENGER CLASS
# ------------------------------------------------------------

survival_class = df.groupby("Pclass")["Survived"].mean() * 100

print("\n========== SURVIVAL RATE BY CLASS ==========")
print(survival_class)

plt.figure(figsize=(8, 5))

survival_class.plot(kind="bar")

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")

plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("visualizations/survival_rate_class.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 17. FINAL SUMMARY
# ------------------------------------------------------------

print("\n================================================")
print("WEEK 2 EDA ANALYSIS COMPLETED SUCCESSFULLY")
print("================================================")

print("\nTotal rows:", len(df))
print("Total columns:", len(df.columns))

print("\nVisualizations saved inside:")
print("visualizations/")

print("\nFiles generated successfully.")