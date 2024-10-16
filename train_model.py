import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt



def train_random_forest(data_loc, n_estimators, max_depth, test_size):
    data = pd.read_csv(data_loc)

    # Load your data
    X = data.drop(columns=['league.season', 'teams.home.id', 'teams.away.id', 'home_spread'])
    y = data['home_spread']  # Target variable

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize and train the model
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Random Forest Model Performance:")
    print(f"n_estimators: {n_estimators}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    # plotting feature importance
    global_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    global_importances.sort_values(ascending=True, inplace=True)
    plt.figure(figsize=(10, 8))
    global_importances.plot.barh(color='green')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Global Feature Importance - Built-in Method")
    plt.savefig("feature_importance.png")






def train_DNN():
    data = pd.read_csv('./formatted_data/training_data.csv')

    X = data.drop(columns=['league.season', 'teams.home.id', 'teams.away.id', 'home_spread'])
    y = data['home_spread']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Define the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)  # Output layer for regression
    ])


    # Compile the model
    model.compile(optimizer='adam', loss='mean_absolute_error')

    # Train the model with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stopping])

    # Evaluate on the test set
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test loss (MAE): {test_loss}')


def main():
    ### RANDOM FOREST GENERATOR ###
    # if depth is 10, MAE plateaus to around 6.44
    # if depth is 5, MAE plateaus to around 6.42
    # RFG is always a little above 6.4 though
    # the most important features are the home and away streaks, followed (not closely) by winning percentages
    '''
    for n_estimators in [10, 25, 50, 100, 200, 500]:
        max_depth = 4
        test_size = 0.2
        train_random_forest('./formatted_data/training_data.csv', n_estimators, max_depth, test_size)
    '''
    train_random_forest('./formatted_data/training_data.csv', 100, 10, 0.2)

    ### DEEP NEURAL NETWORK ###
    train_DNN()


if __name__ == "__main__":
    main()