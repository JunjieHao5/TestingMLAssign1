# Library Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as matplt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score


# Read dataset from github
dataframe = pd.DataFrame(pd.read_excel("https://github.com/JunjieHao5/CS-6375-ML-Assignment1/raw/main/Concrete_Dataset.xlsx", sheet_name = 'Concrete', index_col = 'No'))
print("Dataset Loaded.")

# Check for null values and print the number of null valus
null_count = dataframe.isnull().sum().sum()
print(f"\nNumber of null valus: {null_count}")

# Remove any null values if there is any
if null_count > 0:
    dataframe.dropna(inplace=True)
    print("\nNull entries removed.")

# Check for duplicate rows and print the number of duplicates
duplicate_count = dataframe.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")

# Remove duplicate rows if they exist
if duplicate_count > 0:
    dataframe.drop_duplicates(inplace=True)
    print("\nDuplicate rows removed.")

# Identify categorical columns
categorical_cols = dataframe.select_dtypes(include=['object']).columns
# print(f"\nCategorical columns: {categorical_cols}")

# Convert categorical columns to numerical using one-hot encoding
if len(categorical_cols) > 0:
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols)
    print("\nCategorical columns converted to numerical.")
else:
    print("\nNo categorical columns found.")

# # Display the updated DataFrame dtypes
# print("Updated Data types:")
# print(dataframe.dtypes)

# Other necessary pre-processing 
# Print dataset description to check if there is any suspicious figure
print("\nDescription of the dataframe:")
print(dataframe.describe().round(2))

# Show the correlation of the attributes to the Concrete compressive strength-Mpa
print("\nCorrelation between all attributes and Concrete compressive strength-Mpa:")
print(abs(dataframe.corr())['Y Concrete compressive strength-Mpa'].sort_values(ascending = False).round(2))

# No attribute is not suitable or is not correlated, we don't remove any.

# Store dataset into the XY matrix
X = np.array(dataframe.drop(['Y Concrete compressive strength-Mpa'], axis = 1))
Y = np.array(dataframe['Y Concrete compressive strength-Mpa'])

# Split dataset into train and test data by a ratio of 80/20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 99)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Scale the data set
std_scaler = StandardScaler()
std_scaler.fit(X_train)
X_train_scaled = std_scaler.transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

# Train model with different set of learning rate and number of iterations 
# Set learning rate and number of iterations in arrays
train_learning_rates = [0.001, 0.01, 0.1, 0.3, 0.5, 1]
train_iterations = [100, 500, 100, 200, 400, 800, 1000]
config_count = 1

# Store evaluation statistics mse, r2, and explained variance into arrays
mse_arr = []
r2_arr = []
ev_arr = []

# Open Log File
log_file = open("part2_log.txt","w")

for i in train_learning_rates:
    for j in train_iterations:   
        # Learning Rate as i and the number of iterations as j
        lrsgd = SGDRegressor(eta0 = i, max_iter = j, random_state = 99)
        lrsgd.fit(X_train_scaled, np.array(Y_train))
        Y_pred = lrsgd.predict(X_test_scaled)
        
        # Computer the evaluation static
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        ev = explained_variance_score(Y_test, Y_pred)
        
        # Store Results
        mse_arr.append(mse)
        r2_arr.append(r2)
        ev_arr.append(ev)
        
        # Write them into the log file
        log_file.write("Run: {} || MSE: {:.2f} || R^2 Score: {:.2f} || Explained Variance Score: {:.2f} || Learning Rate: {} || Iterations: {}\n".format(
            config_count, mse, r2, ev, i, j))
        config_count += 1

best_param_idx = mse_arr.index(min(mse_arr))
print("\nBest Performance Run:", best_param_idx,"th")
print("MSE: {:.2f}".format(mse_arr[best_param_idx]))
print("R^2: {:.2f}".format(r2_arr[best_param_idx]))
print("Explained Variance: {:.2f}".format(ev_arr[best_param_idx]))
print("Learning Rate:", train_learning_rates[(best_param_idx % len(train_learning_rates))])
print("Iterations:", train_iterations[(best_param_idx % len(train_iterations))])

log_file.close()

#Train model with cikit learn parameters

# Training model
print("\nTraining model with default library parameters:")
lrsgd = SGDRegressor()
lrsgd.fit(X_train_scaled, np.array(Y_train))
Y_pred = lrsgd.predict(X_test_scaled)

# Computer evalaation static
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
ev = explained_variance_score(Y_test, Y_pred)
        

print("MSE: {:.2f}".format(mse))
print("R^2 Score: {:.2f}".format(r2))
print("Explained Variance Score: {:.2f}".format(ev))


# Show the best linear regression model equation
coefficients = lrsgd.coef_
intercept = lrsgd.intercept_

# Create the equation string
equation = "y = {:.2f}".format(intercept[0])
for i, coef in enumerate(coefficients):
    equation += " + {:.2f}x{}".format(coef, i + 1)

print("\nThe best linear regression model is:")
print(equation)

