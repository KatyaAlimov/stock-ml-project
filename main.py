import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub 
import os
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import Dense, Dropout, Input 

# Load the dataset
path = kagglehub.dataset_download("asadullahcreative/us-stock-market-historical-ohlcv-dataset")
data = os.path.join(path, "stock_prices_daily.csv")
df = pd.read_csv(data)

df['Date'] = pd.to_datetime(df['Date'], utc=True)
#sort so future shift is acurate 

df = df.sort_values(['Ticker', 'Date'])
#processing , adjsuting to date time format, sorting by chronological order

df['Target'] = (df.groupby('Ticker')['Close'].shift(-5) - df['Close']) / df['Close']

# creating target by shifting close price by 5 days) Predicting next week price 
df['Target'] = df['Target'].clip(-0.2, 0.2)
# Dropping rows with NaN values in the target column
df = df.dropna()

# defining input feature and target variable
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = df[features]
y = df['Target']

# Splitting the data into training and testing sets 80 % and 20 % respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
# calculating mean and standard deviation of the training data for scaling
X_train_scaled = scaler.fit_transform(X_train)
#transforming the testing data using the same training parameters to prevent data leakage
X_test_scaled = scaler.transform(X_test)

print("Training Linear Regression Baseline Model")
# Initializing and training the linear regression model)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
# Predicting on the test set
y_pred = model.predict(X_test_scaled)
# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression MSE: {mse}")
print(f"Linear Regression R^2 Score: {r2}")


print("\nParameter Analysis for KNN Regression")
# Testing different values of K for KNN regression
k_val = [3, 5, 11]
knnPreformance = []

for k in k_val:
    print("Training kNN model with k =", k)
    #intitalize kNN with the current k value
    knn = KNeighborsRegressor(n_neighbors=k)
    # find the most similar days int the training data
    knn.fit(X_train_scaled, y_train)

    #predict price based on the avarge of k-nearest historical paterns
    preds = knn.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    # storing result for checking which k is best 

    knnPreformance.append((k, mse, r2, preds))
    print(f"kNN Regression with k={k} MSE: {mse}, R^2 Score: {r2}")

# Extract the tuple with highest R2 score from results list 
best_result = max(knnPreformance, key=lambda x: x[2])
best_k, best_mse, best_r2, best_preds = best_result

print(f"\nBest k value: {best_k} with MSE: {best_mse} and R^2 Score: {best_r2}")

# NN 
print("\nTraining Neural Network Model")

nn = Sequential([
    # input layer + first hidden layer with 64 neurons and sigmoid activation
    Dense(64, activation='sigmoid', input_shape=(X_train_scaled.shape[1],)),
    # dropout layer to prevent overfitting
    Dropout(0.2),
    # second hidden layer with 32 neurons and sigmoid activation
    Dense(32, activation='sigmoid'),
    # output layer with 1 neuron for price prediction
    Dense(1)
])
     
# configure model with adam optimizer and mean squared error loss function
nn.compile(optimizer='adam', loss='mean_squared_error')


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)



# training the model with mini batches of 1024 for speed optimazation 
nn.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=1024,
    callbacks=[early_stopping],
    verbose=1
)
# final prediction and evaluation of the model
nn_final_preds = nn.predict(X_test_scaled).flatten()
nn_mse = mean_squared_error(y_test, nn_final_preds)
nn_r2 = r2_score(y_test, nn_final_preds)
print(f"Neural Network MSE: {nn_mse}")
print(f"Neural Network R^2 Score: {nn_r2}")








# comparing all models via visualization
plt.figure(figsize=(14, 7))

# select the final 100 timesteps for clear, readable visualization
data_slice = df['Date'].iloc[-100:]

plt.plot(data_slice, y_test.iloc[-100:], label='Actual Price', color='blue', linewidth=2)
plt.plot(data_slice, best_preds[-100:], label=f'KNN Predictions (k={best_k})', color='green', linestyle='--')
plt.plot(data_slice, nn_final_preds[-100:], label='Neural Network Predictions', color='pink', linestyle='--')

# formatting the final plot
plt.title('Stock Price Prediction Comparison - Next Week Asset Returns')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
# Display the comparison plot
plt.show()

# 1. Prepare Experimental Data
df_comperison = df.copy() # Corrected from pd.copy()
df_comperison['SMA_5'] = df_comperison.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).mean())
df_comperison['Price_Change'] = df_comperison.groupby('Ticker')['Close'].pct_change()
# Corrected: added closing parenthesis
df_comperison['Volatility'] = df_comperison.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).std())

df_comperison = df_comperison.dropna()
extended_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'Price_Change', 'Volatility']
X_extended = df_comperison[extended_features]
y_extended = df_comperison['Target']

print("\n" + "="*50)
print("Experimental phase: Greedy feature selection")
print("="*50)

# Corrected: removed duplicate random_state
X_sample, _, y_sample, _ = train_test_split(X_extended, y_extended, train_size=5000, shuffle=True, random_state=42)

# Corrected: changed [] to {} for dictionary syntax
selected_features = {
    "linear_model": LinearRegression(),
    "rf_model": RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
}

# Run the loop
for name, model_obj in selected_features.items():
    print(f"\nEvaluating feature importance with {name}...")
    sfs = SequentialFeatureSelector(model_obj, n_features_to_select=5, direction='forward', cv=3)
    sfs.fit(X_sample, y_sample)
    selected_cols = X_sample.columns[sfs.get_support()]
    print(f"Selected features for {name}: {list(selected_cols)}")



print(f"\nProceeding with dynamically selected features: {list(selected_cols)}")
### trining the linear regression model with the selected features
# 1. Use the same filtered data from your experimental phase
# (df_comperison has the technical indicators like Volatility and Price_Change)
X_fs_lin = df_comperison[selected_cols] # No more hard-coded strings
y_fs_lin = df_comperison['Target']

X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_fs_lin, y_fs_lin, test_size=0.2, shuffle=False)

scaler_lin = StandardScaler()
X_train_lin_scaled = scaler_lin.fit_transform(X_train_lin)
X_test_lin_scaled = scaler_lin.transform(X_test_lin)

lin_reg_fs = LinearRegression()
lin_reg_fs.fit(X_train_lin_scaled, y_train_lin)

lin_fs_preds = lin_reg_fs.predict(X_test_lin_scaled)
lin_fs_r2 = r2_score(y_test_lin, lin_fs_preds)

print("\n" + "="*40)
print("LINEAR REGRESSION: DYNAMIC FEATURES")
print(f"Features used: {list(selected_cols)}")
print(f"R2 Score:      {lin_fs_r2:.4f}")
print("="*40)

# 1. Prepare data with RF Selected Features
selected_features = ['Low', 'Close', 'Volume', 'Price_Change', 'Volatility']
X_fs = df_comperison[selected_features]
y_fs = df_comperison['Target']

# Split (using shuffle=False for time-series integrity)
X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_fs, y_fs, test_size=0.2, shuffle=False)

# Scale
scaler_fs = StandardScaler()
X_train_fs_scaled = scaler_fs.fit_transform(X_train_fs)
X_test_fs_scaled = scaler_fs.transform(X_test_fs)

# 2. Build the Model (Updated input shape and Activation)
# 2. Build the Model (Updated input shape and Activation)
nn_selected = Sequential([
    Input(shape=(X_train_fs_scaled.shape[1],)), # Line 227: The modern way to define input
    Dense(64, activation='relu'), 
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

nn_selected.compile(optimizer='adam', loss='mean_squared_error')

print("\n--- Training NN with Selected Features ---")
# 3. Train
nn_selected.fit(
    X_train_fs_scaled, y_train_fs,
    validation_split=0.2,
    epochs=100,
    batch_size=1024,
    callbacks=[early_stopping],
    verbose=1
)

# 4. Final Metrics
fs_preds = nn_selected.predict(X_test_fs_scaled).flatten()
fs_mse = mean_squared_error(y_test_fs, fs_preds)
fs_r2 = r2_score(y_test_fs, fs_preds)

print("\n" + "="*40)
print("RESULTS: SELECTED FEATURES ONLY")
print(f"MSE: {fs_mse:.4f}")
print(f"R2 Score: {fs_r2:.4f}")
print("="*40)


plt.figure(figsize=(15, 6))

# We'll look at the last 100 days of the test set
test_dates = df_comperison['Date'].iloc[-100:]
actual_returns = y_test_fs.iloc[-100:]
predicted_returns = fs_preds[-100:]

plt.plot(test_dates, actual_returns, label='Actual 5-Day Return', color='#1f77b4', alpha=0.7, linewidth=2)
plt.plot(test_dates, predicted_returns, label='NN Predicted Return', color='#ff7f0e', linestyle='--', linewidth=2)

# Add a reference line at 0 (No change)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.title('Neural Network: Actual vs. Predicted Returns (Post-Feature Selection)')
plt.xlabel('Date')
plt.ylabel('Return % (Decimal Form)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 8))

plt.scatter(y_test_fs, fs_preds, alpha=0.3, color='purple')

# Perfect prediction line
lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label="Perfect Prediction")

plt.title('Actual vs. Predicted Scatter')
plt.xlabel('Actual Returns')
plt.ylabel('Predicted Returns')
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()