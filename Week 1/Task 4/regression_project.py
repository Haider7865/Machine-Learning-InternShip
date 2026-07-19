import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score, learning_curve

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

# Load Dataset
df = pd.read_csv("student_performance.csv")

print(df.head())

# Remove Duplicate Rows
df = df.drop_duplicates()

# Check Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Independent and Dependent Variables
X = df[["Study_Hours","Attendance","Assignment_Score"]]
y = df["Exam_Score"]

# Split Dataset
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Baseline Model
baseline = LinearRegression()
baseline.fit(X_train,y_train)

base_pred = baseline.predict(X_test)

print("\nBaseline MAE:",
      round(mean_absolute_error(y_test,base_pred),2))

# Adjusted R2 Function
def adjusted_r2(r2,n,p):
    return 1-((1-r2)*(n-1)/(n-p-1))

results=[]

# Regression Models
models={
"Linear Regression":LinearRegression(),
"Polynomial Regression":make_pipeline(
PolynomialFeatures(2),
LinearRegression()
),
"Ridge Regression":Ridge(),
"Lasso Regression":Lasso(alpha=0.1),
"Elastic Net":ElasticNet(alpha=0.1,l1_ratio=0.5),
"Decision Tree":DecisionTreeRegressor(random_state=42),
"Random Forest":RandomForestRegressor(random_state=42),
"Gradient Boosting":GradientBoostingRegressor(random_state=42)
}

# Function to Train and Evaluate Models
def evaluate(model,name):

    model.fit(X_train,y_train)
    pred=model.predict(X_test)

    mae=mean_absolute_error(y_test,pred)
    mse=mean_squared_error(y_test,pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_test,pred)
    adj=adjusted_r2(r2,len(y_test),X.shape[1])
    mape=mean_absolute_percentage_error(y_test,pred)

    kfold=KFold(n_splits=5,shuffle=True,random_state=42)
    cv=cross_val_score(model,X,y,cv=kfold,scoring="r2").mean()

    repeat=RepeatedKFold(n_splits=5,n_repeats=2,random_state=42)
    rcv=cross_val_score(model,X,y,cv=repeat,scoring="r2").mean()

    results.append([
        name,
        round(mae,2),
        round(mse,2),
        round(rmse,2),
        round(r2,2),
        round(adj,2),
        round(mape,2),
        round(cv,2),
        round(rcv,2)
    ])

    return pred

# Train All Models
for name,model in models.items():
    print("\n",name)
    prediction=evaluate(model,name)

# Model Comparison Table
comparison=pd.DataFrame(results,columns=[
"Model",
"MAE",
"MSE",
"RMSE",
"R2",
"Adjusted R2",
"MAPE",
"KFold",
"Repeated CV"
])

print("\n========== Model Comparison ==========")
print(comparison)

# Best Model
best=comparison.loc[comparison["R2"].idxmax()]

print("\n========== Best Model ==========")
print(best)

# Save Best Prediction
best_prediction=prediction

# Residual Plot
residual=y_test-best_prediction

plt.figure(figsize=(6,4))
plt.scatter(best_prediction,residual)
plt.axhline(y=0,color="red")
plt.title("Residual Plot")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.show()

# Learning Curve
size,train,test=learning_curve(
RandomForestRegressor(random_state=42),
X,y,cv=5)

plt.figure(figsize=(6,4))
plt.plot(size,train.mean(axis=1),label="Training")
plt.plot(size,test.mean(axis=1),label="Validation")
plt.title("Learning Curve")
plt.xlabel("Training Samples")
plt.ylabel("Score")
plt.legend()
plt.show()

# Linear Relationship
plt.figure(figsize=(6,4))
plt.scatter(df["Study_Hours"],df["Exam_Score"])
plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.show()

# Correlation Matrix
print("\nCorrelation Matrix")
print(X.corr())

# Regression Assumptions
print("\nRegression Assumptions")
print("- Linear Relationship")
print("- Independent Observations")
print("- Normal Residuals")
print("- Constant Variance")
print("- Low Multicollinearity")

# Bias and Variance
print("\nBias : Model is too simple.")
print("Variance : Model learns too much from training data.")

# Overfitting and Underfitting
print("\nOverfitting : High training performance but poor testing performance.")
print("Underfitting : Poor performance on both training and testing data.")

# Error Analysis
print("\nError Analysis")
print("Lower MAE, MSE and RMSE indicate better predictions.")
print("Higher R2 and Adjusted R2 indicate better model performance.")
print("Lower MAPE means prediction error is small.")

# Final Recommendation
print("\nRecommended Model :",best["Model"])
print("Reason : Highest R2 Score with good Cross Validation performance.")

# Limitations
print("\nLimitations")
print("- Small dataset used")
print("- More features can improve accuracy")
print("- XGBoost and LightGBM are only discussed in theory")

print("\n========== PROJECT COMPLETED ==========")