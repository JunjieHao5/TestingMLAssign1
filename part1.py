# Library Imports
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as matplt

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


# All plot are at he bottom, according to the plot and all pre-processing steps,
# no attribute is not suitable or is not correlated, we don't remove any.

# Define Linaear Regression method without using Scikit Learn package
class LinearRegression:
    # Initialize the learning rate and the max Iteration
    def __init__(self, learning_rate, max_iterations):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    # Find the best fit line
    def bestfit(self, X, Y):
        self.X = X
        self.Y = Y

        # Create an array for the loss function
        self.cost = []

        # Make a new columns to hold X0 as 1
        self.weights = np.random.rand(self.X.shape[1] + 1).reshape(1, -1) 

        # Create the weights from w0 to w8 for our dataset             
        self.new_X = np.insert(self.X.T, 0, np.ones(self.X.shape[0]), axis=0)  

        # Set Cost function to compute the errors later 
        derivation_cost = 0
        
        while self.max_iterations > -1:
            # Compute the predicted value using woxo + w1x1 + w2x2 + ... + w8x8
            self.hypothesis = np.dot(self.weights, self.new_X)

            # Store into the cost array
            self.cost.append(self.cost_function(self.hypothesis, Y))

            # Find the number of rows in X
            m = self.X.shape[0] 

            # Take derivative of the function
            derivation_cost = (self.new_X@(self.hypothesis - self.Y).T) * 1 / m

            # Theta = Theta - learning rate * derivative
            self.weights -= (self.learning_rate * derivation_cost.reshape(1, -1))
            self.max_iterations -= 1

    # Find the cost value     
    def cost_function(self, X, Y):
        loss = np.sum(np.square(X.reshape(-1, 1) - Y.reshape(-1,1))) / (2 * X.shape[0])
        return np.round(loss, 3)
    
    # Define the r2_value method
    def r2_score(self,X,Y):
        return 1 - (((Y - self.predict(X)) ** 2).sum() / ((Y - Y.mean()) ** 2).sum())
    
    # Defube the predict method to find the weights for all the X 
    def predict(self,X):
        X = np.insert(X.T, 0, np.ones(X.shape[0]), axis = 0)
        return np.dot(self.weights, X)
    
# Store dataset into the XY matrix
X = np.array(dataframe.drop(['Y Concrete compressive strength-Mpa'], axis = 1))
Y = np.array(dataframe['Y Concrete compressive strength-Mpa'])

# Split dataset into train and test data by a ratio of 80/20
def train_test_split_manual(X, Y, test_size=0.2, random_state=None):
    if random_state:
        random.seed(random_state)
    
    # Zip X and Y together
    combined = list(zip(X, Y))
    
    # Shuffle the data
    random.shuffle(combined)
    
    # Unzip the shuffled data
    X[:], Y[:] = zip(*combined)
    
    # Split the data
    split_index = int(len(X) * (1 - test_size))
    X_train = X[:split_index]
    X_test = X[split_index:]
    Y_train = Y[:split_index]
    Y_test = Y[split_index:]
    
    return X_train, X_test, Y_train, Y_test

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split_manual(X, Y, test_size=0.2, random_state=99)

# Implemente standard scaling manually
def standard_scaler(X):
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    return (X - mean) / std_dev, mean, std_dev

# Apply the scaling to the test set based on the training set statistics
def apply_standard_scaler(X, mean, std_dev):
    return (X - mean) / std_dev

# Convert lists to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Scale the training set and getting mean and std_dev
X_train_scaled, mean, std_dev = standard_scaler(X_train)

# Scale the test set using the same mean and std_dev
X_test_scaled = apply_standard_scaler(X_test, mean, std_dev)

# print("X_train_scaled:", X_train_scaled)
# print("X_test_scaled:", X_test_scaled)

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
log_file = open("part1_log.txt","w")

for i in train_learning_rates:
    for j in train_iterations:
        # Keep learning rate as i and the number of iterations as j
        linear_regressor = LinearRegression(learning_rate = i, max_iterations = j)
        linear_regressor.bestfit(X_train_scaled, np.array(Y_train))
        Y_pred = linear_regressor.predict(X_test_scaled)

        # To calculate Mean Squared Error
        def mean_squared_error(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        # To calculate R-squared 
        def r2_score(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)

        # To calculate Explained Variance 
        def explained_variance_score(y_true, y_pred):
            var_res = np.var(y_true - y_pred)
            var_true = np.var(y_true)
            return 1 - (var_res / var_true)

        # Compute evaluation statistics
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        ev = explained_variance_score(Y_test, Y_pred)
        
        # Store The results
        mse_arr.append(mse)
        r2_arr.append(r2)
        ev_arr.append(ev)
        
        # Write them into the log file 
        log_file.write("Run: {} || MSE: {:.2f} || R^2 Score: {:.2f} || Explained Variance Score: {:.2f} || Learning Rate: {} || Iterations: {}\n".format(
            config_count, mse, r2, ev, i, j))
        config_count += 1

# Find the best iteratin with the best proformance (lowest MSE)
best_param_idx = mse_arr.index(min(mse_arr))
print("\nBest Performance Run:", best_param_idx,"th")
print("MSE: {:.2f}".format(mse_arr[best_param_idx]))
print("R^2: {:.2f}".format(r2_arr[best_param_idx]))
print("Explained Variance: {:.2f}".format(ev_arr[best_param_idx]))
print("Learning Rate:", train_learning_rates[(best_param_idx % len(train_learning_rates))])
print("Iterations:", train_iterations[(best_param_idx % len(train_iterations))])

# Create an instance of LinearRegression with specific parameters here 0.3 learning rate and 1000 interation
lr = LinearRegression(learning_rate=0.5, max_iterations=2000)

# Fit the model to your training data
lr.bestfit(X_train_scaled, Y_train)

# Retrieve the weights
best_model_weights = lr.weights.flatten()

# Display the best-fitting linear regression model
equation_str = "y = {:.2f}".format(best_model_weights[0])  # Initialize with w0 (Intercept)
for i, w in enumerate(best_model_weights[1:]):
    equation_str += " + {:.2f}x{}".format(w, i + 1)

print("\nBest linear regression model:")
print(equation_str)

log_file.close()


# # Plots that generate for assignment 1 report, you can find them in the report, or uncomment them at here to see the actual plots

# #Show plots for corraletion of each variable on Concrete compressive strength-Mpa
# sns.set(rc = {'figure.figsize':(14,8)})
# hmap = sns.heatmap(dataframe.corr(), vmin = -1, vmax = 1)
# columns = dataframe.columns

# for i in range(len(columns) - 1):
#     matplt.figure(i)
#     sns.scatterplot(x = columns[i], y = 'Y Concrete compressive strength-Mpa', data = dataframe)

# #Show plots betweent loss and different Learning Rates under 1000 interations
# # 0.01
# linear_regressor = LinearRegression(learning_rate = 0.01, max_iterations = 1000)
# linear_regressor.bestfit(X_train_scaled, np.array(Y_train))
# Y_pred = linear_regressor.predict(X_test_scaled)

# loss = list(linear_regressor.cost)
# matplt.plot(loss)
# matplt.xlabel("Iterations")
# matplt.ylabel("Loss")
# matplt.title("Learning Rate: 0.01")
# matplt.show()

# # 0.1
# linear_regressor = LinearRegression(learning_rate = 0.1, max_iterations = 1000)
# linear_regressor.bestfit(X_train_scaled, np.array(Y_train))
# Y_pred = linear_regressor.predict(X_test_scaled)

# loss = list(linear_regressor.cost)
# matplt.plot(loss)
# matplt.xlabel("Iterations")
# matplt.ylabel("Loss")
# matplt.title("Learning Rate: 0.1")
# matplt.show()

# # 0.3
# linear_regressor = LinearRegression(learning_rate = 0.3, max_iterations = 1000)
# linear_regressor.bestfit(X_train_scaled, np.array(Y_train))
# Y_pred = linear_regressor.predict(X_test_scaled)

# loss = list(linear_regressor.cost)
# matplt.plot(loss)
# matplt.xlabel("Iterations")
# matplt.ylabel("Loss")
# matplt.title("Learning Rate: 0.3")
# matplt.show()

# # 0.5
# linear_regressor = LinearRegression(learning_rate = 0.5, max_iterations = 1000)
# linear_regressor.bestfit(X_train_scaled, np.array(Y_train))
# Y_pred = linear_regressor.predict(X_test_scaled)

# loss = list(linear_regressor.cost)
# matplt.plot(loss)
# matplt.xlabel("Iterations")
# matplt.ylabel("Loss")
# matplt.title("Learning Rate: 0.5")
# matplt.show()

# # Plots showing relatinship between MSE and number of interations
# # Update MSE and bias into the Linear Regression method to show the corralation
# class LinearRegression:
#     def __init__(self, learning_rate=0.01, max_iterations=1000):
#         self.learning_rate = learning_rate
#         self.max_iterations = max_iterations
#         self.weights = None
#         self.bias = None
#         self.cost = []  # To store MSE at each iteration

#     def bestfit(self, X, y):
#         num_samples, num_features = X.shape
#         self.weights = np.zeros(num_features)
#         self.bias = 0

#         # Gradient descent
#         for i in range(self.max_iterations):
#             y_pred = np.dot(X, self.weights) + self.bias

#             # Compute and store MSE
#             mse = np.mean((y - y_pred) ** 2)
#             self.cost.append(mse)

#             # Compute gradients
#             dw = -(2 / num_samples) * np.dot(X.T, (y - y_pred))
#             db = -(2 / num_samples) * np.sum(y - y_pred)

#             # Update parameters
#             self.weights -= self.learning_rate * dw
#             self.bias -= self.learning_rate * db

#     def predict(self, X):
#         return np.dot(X, self.weights) + self.bias

# # Run linear regressor 10 interations
# linear_regressor = LinearRegression(learning_rate=0.1, max_iterations=10)
# linear_regressor.bestfit(X_train_scaled, np.array(Y_train))
# Y_pred = linear_regressor.predict(X_test_scaled) 
# #Plot MSE against number of iterations
# matplt.plot(linear_regressor.cost)
# matplt.xlabel("Iterations")
# matplt.ylabel("MSE")
# matplt.title("Learning Rate: 0.1")
# matplt.show()

# # Run linear regressor 50 interations
# linear_regressor = LinearRegression(learning_rate=0.1, max_iterations=50)
# linear_regressor.bestfit(X_train_scaled, np.array(Y_train))
# Y_pred = linear_regressor.predict(X_test_scaled) 
# #Plot MSE against number of iterations
# matplt.plot(linear_regressor.cost)
# matplt.xlabel("Iterations")
# matplt.ylabel("MSE")
# matplt.title("Learning Rate: 0.1")
# matplt.show()

# # Run linear regressor 200 interations
# linear_regressor = LinearRegression(learning_rate=0.1, max_iterations=200)
# linear_regressor.bestfit(X_train_scaled, np.array(Y_train))
# Y_pred = linear_regressor.predict(X_test_scaled) 
# # Plot MSE against number of iterations
# matplt.plot(linear_regressor.cost)
# matplt.xlabel("Iterations")
# matplt.ylabel("MSE")
# matplt.title("Learning Rate: 0.1")
# matplt.show()

# # Run linear regressor 1000 interations
# linear_regressor = LinearRegression(learning_rate=0.1, max_iterations=1000)
# linear_regressor.bestfit(X_train_scaled, np.array(Y_train))
# Y_pred = linear_regressor.predict(X_test_scaled) 
# #Plot MSE against number of iterations
# matplt.plot(linear_regressor.cost)
# matplt.xlabel("Iterations")
# matplt.ylabel("MSE")
# matplt.title("Learning Rate: 0.1")
# matplt.show()