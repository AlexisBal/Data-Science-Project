import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


"""
Preliminary analysis : descriptive statistics
"""
print("\nI) Preliminary analysis : descriptive statistics\n")

# Load dataset into Pandas DataFrame
data = pd.read_csv("garments_worker_productivity.csv")

# How many observations are there ?
nbObservations = data.shape[0]
print("Observations Number: {}\n".format(nbObservations))

# How many variables?
nbVariables = data.shape[1]
print("Variables Number: {}\n".format(nbVariables))

# how many observations contain missing values ?
numberObservationsMissingValues = data.isnull().sum().sum()
print(
    "Observations contain missing values Number: {}\n".format(
        numberObservationsMissingValues
    )
)

# Which variables are concerned with this missing data ?
variablesMissingValues = data.loc[:, data.isnull().any()].columns[0]
print("Variables concerned with this missing data: {}\n".format(variablesMissingValues))

# Drop Rows with missing value
cleanData = data.dropna()

# Calculate descriptive statistics for the target variable actual_productivity
descriptiveStatistics = cleanData["actual_productivity"].describe()
print("Descriptive statistics:\n{}\n".format(descriptiveStatistics))
# Correlation coefficient between actual_productivity and each of the other variables
targetedProductivityCorrelation = cleanData.corr().actual_productivity
print("Correlation coefficient:\n{}\n".format(targetedProductivityCorrelation))


"""
Principal Component Analysis (PCA)
"""
print("\n\nII) Principal Component Analysis (PCA)\n")

#Calculate the variance of each variable 
variance = np.var(cleanData)
print("\n Variance :")
print(variance)

# Manually standardize the variables before performing PCA
cleanData_std = StandardScaler().fit_transform(
    cleanData[
        [
            "targeted_productivity",
            "smv",
            "over_time",
            "incentive",
            "no_of_workers",
            "actual_productivity",
        ]
    ]
)

# Perform PCA using the following variables : targeted_productivity, smv, over_time, incentive, no_of_workers and actual_productivity
pca = PCA().fit(cleanData_std)

# Calculate the loading vectors
loadings = pca.components_
print("\nLoading vectors:\n{}\n".format(loadings))

# Use a biplot with a correlation circle to display both the principal component scores and the loading vectors in a single plot
fig, axis = plt.subplots(figsize=(5, 5))
axis.set_xlim(-1, 1)
axis.set_ylim(-1, 1)
plt.plot([-1, 1], [0, 0], color="silver", linestyle="-", linewidth=1)
plt.plot([0, 0], [-1, 1], color="silver", linestyle="-", linewidth=1)
for j in range(0, 6):
    plt.arrow(
        0, 0, loadings[j, 0], loadings[j, 1], head_width=0.02, width=0.001, color="red"
    )
    plt.annotate(
        cleanData[
            [
                "targeted_productivity",
                "smv",
                "over_time",
                "incentive",
                "no_of_workers",
                "actual_productivity",
            ]
        ].columns[j],
        (loadings[j, 0], loadings[j, 1]),
    )
cercle = plt.Circle((0, 0), 1, color="blue", fill=False)
axis.add_artist(cercle)
plt.title("Correlation Circle")

# Calculate the percentage of variance explained (PVE) by each component
plt.figure(2)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
print("Percentage of variance explained:\n{}\n".format(per_var))
labels = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
plt.plot([1, 2, 3, 4, 5, 6], per_var)
plt.ylabel("Percentage of variance explained (PVE)")
plt.xlabel("Principal component")
plt.title("PVE explained by each component")


"""
Simple Linear Regression Model
"""
print("\n\nIII) Simple Linear Regression Model\n")

# Linear regression model : actual_productivity = β0 + β1incentive + ε
x_train = np.array(cleanData.incentive).reshape((-1, 1))
y_train = np.array(cleanData.actual_productivity)
regressor_1 = LinearRegression()
regressor_1.fit(x_train, y_train)
X = sm.add_constant(x_train)
ols = sm.OLS(y_train, X)
ols_result = ols.fit()
print(ols_result.summary())

# Coefficient estimates
print(
    "Coefficient estimates:\nβ0 = %s\nβ1 = %s\n"
    % (regressor_1.intercept_, regressor_1.coef_[0])
)

# Coefficient of determination (𝑅²)
r_sq = regressor_1.score(x_train, y_train)
print("Coefficient of determination:", r_sq)

# Prediction the productivity
y_pred = regressor_1.predict(x_train)
print("\nPrediction the productivity:", y_pred, sep="\n")


"""
Multiple Linear Regression Model
"""
print("\n\n\nIV) Multiple Linear Regression Model\n")

# Fit linear regression model and return RSS and R squared values
def fit_linear_reg(X, Y):
    regressor_2 = LinearRegression(fit_intercept=True)
    regressor_2.fit(X, Y)
    # Residual Sum of Squares
    RSS = mean_squared_error(Y, regressor_2.predict(X)) * len(Y)
    # Adjusted coefficient of determination
    R = 1 - (1 - regressor_2.score(X, Y)) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)
    return RSS, R


# Initialization variables
Y = cleanData.actual_productivity
X = cleanData[
    ["targeted_productivity", "smv", "wip", "over_time", "incentive", "no_of_workers"]
]
RSS_list, R_list, feature_list, nb_features = [], [], [], []

# Looping over k = 1 to k = 6 features in X
for k in range(1, len(X.columns) + 1):
    # Looping over all possible combinations
    for combo in itertools.combinations(X.columns, k):
        tmp_result = fit_linear_reg(X[list(combo)], Y)
        RSS_list.append(tmp_result[0])
        R_list.append(tmp_result[1])
        feature_list.append(combo)
        nb_features.append(len(combo))

# Append lists
df = pd.DataFrame(
    {
        "numb_features": nb_features,
        "RSS": RSS_list,
        "R_squared": R_list,
        "Features": feature_list,
    }
)

# Best Subset Selection with RSS
df_min = df[df.groupby("numb_features")["RSS"].transform(min) == df["RSS"]]
# Best Subset Selection with adjusted coefficient of determination R^2
df_max = df[df.groupby("numb_features")["R_squared"].transform(max) == df["R_squared"]]
print("Best Subset Selection :")
print(df_min.head(3))

# Plot the curve R^2 versus the number of features
df["max_R_squared"] = df.groupby("numb_features")["R_squared"].transform(max)
fig = plt.figure(figsize=(16, 6))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df.numb_features, df.R_squared, alpha=0.2, color="darkblue")
ax.plot(df.numb_features, df.max_R_squared, color="r", label="Best subset")
ax.set_xlabel("Number of features")
ax.set_ylabel("R^2")
ax.set_title("R^2 - Best subset selection")
ax.legend()

# Linear regression model with the selected model
y_train_bis = cleanData.actual_productivity
x_train_bis = cleanData[["targeted_productivity", "smv", "incentive"]]
regressor_bis = LinearRegression()
regressor_bis.fit(x_train_bis, y_train_bis)
X_bis = sm.add_constant(x_train_bis)
ols = sm.OLS(y_train_bis, X_bis)
ols_result = ols.fit()
print(ols_result.summary())

# Coefficient estimates
print("\nCoefficient estimates:\nβ0 =", regressor_bis.intercept_)
for x in range(1, len(regressor_bis.coef_) + 1):
    print("β%s = %s" % (x, regressor_bis.coef_[x - 1]))

# Coefficient of determination (𝑅²)
r_sq_bis = regressor_bis.score(x_train_bis, y_train_bis)
print("\nCoefficient of determination:", r_sq_bis)

# Prediction the productivity
y_pred_bis = regressor_bis.predict(x_train_bis)
print("\nPrediction the productivity:", y_pred_bis, sep="\n")

# Plot figures
plt.show()
