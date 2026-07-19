import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load Dataset
df = pd.read_csv("student_performance.csv")
print("========== Student Dataset ==========")
print(df)

# Task 1: Descriptive Statistics
print("\nTask 1: Descriptive Statistics")

mean = df["Exam_Score"].mean()
median = df["Exam_Score"].median()
mode = df["Exam_Score"].mode()[0]
variance = df["Exam_Score"].var()
std = df["Exam_Score"].std()

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Variance:", variance)
print("Standard Deviation:", std)

# Task 2: Percentiles
print("\nTask 2: Percentiles")

q1 = np.percentile(df["Exam_Score"],25)
q2 = np.percentile(df["Exam_Score"],50)
q3 = np.percentile(df["Exam_Score"],75)

minimum = df["Exam_Score"].min()
maximum = df["Exam_Score"].max()

print("25th Percentile:", q1)
print("50th Percentile:", q2)
print("75th Percentile:", q3)
print("Minimum:", minimum)
print("Maximum:", maximum)

print("Middle 50% Score Range:", q1, "to", q3)

# Task 3: Correlation
print("\nTask 3: Correlation")

corr1 = df["Study_Hours"].corr(df["Exam_Score"])
corr2 = df["Attendance"].corr(df["Exam_Score"])
corr3 = df["Assignment_Score"].corr(df["Exam_Score"])

print("Study Hours & Exam Score:", corr1)
print("Attendance & Exam Score:", corr2)
print("Assignment Score & Exam Score:", corr3)

# Scatter Plot
plt.scatter(df["Assignment_Score"], df["Exam_Score"])
plt.title("Assignment Score vs Exam Score")
plt.xlabel("Assignment Score")
plt.ylabel("Exam Score")
plt.show()

# Task 4: Probability
print("\nTask 4: Probability")
total = len(df)

pass_students = len(df[df["Exam_Score"] >= 50])
score80 = len(df[df["Exam_Score"] >= 80])
study5 = len(df[df["Study_Hours"] > 5])

print("Probability of Passing:", pass_students/total)
print("Probability of Score >=80:", score80/total)
print("Probability of Study Hours >5:", study5/total)

# Task 5: Normal Distribution
print("\nTask 5: Normal Distribution")

plt.hist(df["Exam_Score"])
plt.title("Histogram of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Students")
plt.show()

df["Z_Score"] = stats.zscore(df["Exam_Score"])

print(df[["Student_ID","Exam_Score","Z_Score"]])

print("\nStudents with |Z Score| > 2")

print(df[abs(df["Z_Score"]) > 2])

# Task 6: Outlier Detection
print("\nTask 6: Outlier Detection")

IQR = q3 - q1

Lower_Bound = q1 - (1.5 * IQR)
Upper_Bound = q3 + (1.5 * IQR)

print("Lower Bound:", Lower_Bound)
print("Upper Bound:", Upper_Bound)

outliers = df[(df["Exam_Score"] < Lower_Bound) | (df["Exam_Score"] > Upper_Bound)]

print("\nOutliers")

print(outliers)

plt.boxplot(df["Exam_Score"])
plt.title("Box Plot of Exam Scores")
plt.show()