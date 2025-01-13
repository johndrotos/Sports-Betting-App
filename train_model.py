import pandas as pd
import numpy as np
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

import matplotlib.pyplot as plt





def train_random_forest(data_loc, n_estimators, max_depth, test_size):
    data = pd.read_csv(data_loc)

    # Load the data
    X = data.drop(columns=['league.season', 'teams.home.id', 'teams.away.id', 'home_spread', 'winner', 'total'])
    y = data['total']  # Target variable

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize and train the model
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    # Spread: 
    mae = mean_absolute_error(y_test, y_pred)    
    # MoneyLine:
    # y_pred_class = (y_pred > 0).astype(int)  # 1 for home win, 0 for away win
    # accuracy = accuracy_score(y_test, y_pred_class) * 100

    print(f"Random Forest Model Performance:")
    print(f"n_estimators: {n_estimators}")
    print(f"MAE: {mae}")

    # plotting feature importance
    global_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    global_importances.sort_values(ascending=True, inplace=True)
    plt.figure(figsize=(18, 14))
    global_importances.plot.barh(color='green')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Global Feature Importance - Built-in Method")
    plt.tight_layout()
    plt.savefig("feature_importance.png")

def train_xgBoost(data_loc, n_estimators, max_depth, test_size):
    data = pd.read_csv(data_loc)

    # Load the data
    X = data.drop(columns=['league.season', 'teams.home.id', 'teams.away.id', 'home_spread', 'winner', 'total', 'id'])
    y = data['total']  # Target variable

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize the XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',  # Use squared error for regression
        n_estimators=n_estimators,              # Number of boosting rounds
        max_depth=max_depth,                   # Maximum tree depth
        learning_rate=0.1,             # Step size shrinkage
        random_state=42
    )
    xgb_model.fit(X_train, y_train)


    # Predict on the test set
    y_pred = xgb_model.predict(X_test)

    # Evaluate the model
    # Spread: 
    mae = mean_absolute_error(y_test, y_pred)    
    # MoneyLine:
    # y_pred_class = (y_pred > 0).astype(int)  # 1 for home win, 0 for away win
    # accuracy = accuracy_score(y_test, y_pred_class) * 100

    print(f"XGBoost Performance:")
    print(f"MAE: {mae}")

    # plotting feature importance
    xgb.plot_importance(xgb_model, importance_type='weight')
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.savefig("XGBoost_feature_importance.png")

def train_DNN():
    data = pd.read_csv('./formatted_data/training_data.csv')
    test_data = data[data['league.season'] == '2023-2024']
    train_data = data[data['league.season'] != '2023-2024']
    
    # Load the data
    X_train = train_data.drop(columns=['league.season', 'teams.home.id', 'teams.away.id', 'home_spread', 'winner', 'total', 'odds_spread'])
    y_train = train_data['home_spread']

    X_test = test_data.drop(columns=['league.season', 'teams.home.id', 'teams.away.id', 'home_spread', 'winner', 'total', 'odds_spread'])
    y_test = test_data['home_spread']
    odds_spread_test = test_data['odds_spread']


    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dense(1)
    ])



    # Compile the model
    optimizer = Adam(learning_rate=0.0005)
    # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    # Train the model with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(
        X_train, 
        y_train, 
        validation_split = 0.2, 
        epochs=300, 
        batch_size=32, 
        callbacks=[early_stopping])

    # Evaluate on the test set
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test loss (MAE): {test_loss}')

    # Predict spreads on the test set
    y_pred = model.predict(X_test)

    # Compare predictions with odds spreads
    test_df = pd.DataFrame({
        'actual_spread': y_test,
        'predicted_spread': y_pred.flatten(),
        'odds_spread': odds_spread_test
    })
    
    test_df['model_error'] = abs(test_df['actual_spread'] - test_df['predicted_spread'])
    test_df['odds_error'] = abs(test_df['actual_spread'] - test_df['odds_spread'])
    avg_model_error = test_df['model_error'].mean()
    avg_odds_error = test_df['odds_error'].mean()
    print(f'Average Model Error (MAE): {avg_model_error:.2f}')
    print(f'Average Odds Spread Error (MAE): {avg_odds_error:.2f}')
    
    test_df.to_csv('testing.csv')

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
    # train_random_forest('./formatted_data/training_data.csv', 100, 10, 0.2)



    ### DEEP NEURAL NETWORK ###
    #train_DNN()


    ### XGBoost ###
    train_xgBoost('./formatted_data/training_data.csv', 100, 6, 0.2)


if __name__ == "__main__":
    main()