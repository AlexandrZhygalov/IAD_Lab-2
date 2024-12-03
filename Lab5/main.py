import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')


"""
Варіант 23
AdaBoostRegressor. Розглянути рiзнi значення параметрiв n_estimators, learning_rate та loss цього алгоритму.

Початковi данi:
www.kaggle.com/rahulsah06/gooogle-stock-price
"""

def visual_dataset(path=r"C:\Users\User\PycharmProjects\L5\Google_Stock_Price_Test.csv"):
    data = pd.read_csv(path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)

    # Вибір ознак і цільової змінної
    X = np.arange(len(data)).reshape(-1, 1)  # Номер дня як змінна
    y = data['Close'].values  # Ціна закриття

    # Візуалізація початкових даних
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], y, label='Google Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Google Stock Price Over Time')
    plt.legend()
    plt.show()
    return X, y


def split_dataset(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_ensemble_models(X_val, y_val, X_test, y_test):
    models = {
        'AdaBoost with DecisionTree': AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), random_state=42),
        'AdaBoost with SVR': AdaBoostRegressor(estimator=SVR(kernel='linear'), random_state=42),
    }

    param_grids = {
        'AdaBoost with DecisionTree': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.5],
            'estimator__max_depth': [3, 5],
        },
        'AdaBoost with SVR': {
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.01, 0.1, 1.0],
            'estimator__C': [0.1, 1.0, 10.0],
            'estimator__gamma': ['scale', 'auto'],
            'estimator__kernel': ['linear', 'rbf']
        },
    }

    best_models = {}
    for name, model in models.items():
        print(f"{name}:")
        grid_search = GridSearchCV(model, param_grids.get(name, {}), cv=3, scoring='r2')
        grid_search.fit(X_val, y_val)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")


    print("\nMetrics:")
    results = {}
    for name, model in best_models.items():
        start_time = time.time()
        y_test_pred = model.predict(X_test)
        elapsed_time = time.time() - start_time
        r2 = r2_score(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mape = mean_absolute_percentage_error(y_test, y_test_pred)
        results[name] = {'R2': r2, 'RMSE': rmse, 'MAPE': mape, 'Time': elapsed_time}
        print(f"{name} - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, Time: {elapsed_time:.4f}s")

    # Оцінки якості для ансамблю
    ensemble_name = 'AdaBoost with DecisionTree'  # Можна змінити на інший ансамбль
    ensemble_model = AdaBoostRegressor(random_state=42)
    ensemble_scores = []
    base_model_scores = []

    # Навчання базової моделі
    base_model = DecisionTreeRegressor(max_depth=3, random_state=42)
    base_model.fit(X_train, y_train)
    base_r2 = r2_score(y_test, base_model.predict(X_test))

    # Збір даних для графіка
    n_estimators_range = [10, 50, 100, 200]
    for n in n_estimators_range:
        ensemble_model.set_params(n_estimators=n)
        ensemble_model.fit(X_train, y_train)
        y_test_pred = ensemble_model.predict(X_test)
        ensemble_scores.append(r2_score(y_test, y_test_pred))
        base_model_scores.append(base_r2)

    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, ensemble_scores, marker='o', label=f'{ensemble_name} R2 Score')
    plt.plot(n_estimators_range, base_model_scores, linestyle='--', label='Base Model R2 Score')
    plt.xlabel('n_estimators')
    plt.ylabel('R2 Score')
    plt.title('R2 Score vs n_estimators')
    plt.legend()
    plt.show()


    # 4. Побудувати графiки на однiй координатнiй площинi (для регресiї)
    # - прогнозiв на основi ансамблю,
    # - прогнозiв на основi окремої моделi base_estimator / estimators,
    # - точок даних з перевiрочної / тестової множин.

    # Прогнози базової моделі
    y_test_pred_base = base_model.predict(X_test)

    # Прогнози ансамблю
    y_test_pred_ensemble = ensemble_model.predict(X_test)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_test, y_test, color='blue', label='True Values', alpha=0.6)
    plt.plot(X_test, y_test_pred_base, color='green', linestyle='--', label='Base Model Predictions')
    plt.plot(X_test, y_test_pred_ensemble, color='red', label=f'AdaBoost Predictions')
    plt.xlabel('Time (days)')
    plt.ylabel('Price')
    plt.title('Comparison of Predictions: Base Model vs Ensemble')
    plt.legend()
    plt.show()


def bias_and_variance(X_train, y_train, X_test, y_test):
    print("\nBias and Variance:")

    n_subsamples = 20
    single_model_predictions = []
    ensemble_predictions = []
    true_values = []
    base_model = DecisionTreeRegressor(max_depth=3, random_state=42)
    ensemble_model = AdaBoostRegressor(random_state=42)

    for _ in range(n_subsamples):
        X_train_sub, y_train_sub = resample(X_train, y_train, random_state=_)

        # Навчання базової моделі
        base_model.fit(X_train_sub, y_train_sub)
        single_model_predictions.append(base_model.predict(X_test))

        # Навчання ансамблю
        ensemble_model.set_params(n_estimators=100)
        ensemble_model.fit(X_train_sub, y_train_sub)
        ensemble_predictions.append(ensemble_model.predict(X_test))

        true_values.append(y_test)


    single_model_predictions = np.array(single_model_predictions)
    ensemble_predictions = np.array(ensemble_predictions)
    true_values = np.array(true_values)

    bias_single = ((single_model_predictions.mean(axis=0) - true_values.mean(axis=0)) ** 2).mean()
    variance_single = single_model_predictions.var(axis=0).mean()
    bias_ensemble = ((ensemble_predictions.mean(axis=0) - true_values.mean(axis=0)) ** 2).mean()
    variance_ensemble = ensemble_predictions.var(axis=0).mean()

    print(f"Single Model - Bias: {bias_single:.4f}, Variance: {variance_single:.4f}")
    print(f"Ensemble - Bias: {bias_ensemble:.4f}, Variance: {variance_ensemble:.4f}")


def timing(X_train, y_train):
    print("\nTiming:")
    base_model = DecisionTreeRegressor(max_depth=3, random_state=42)
    ensemble_model = AdaBoostRegressor(random_state=42)

    start_time = time.time()
    base_model.fit(X_train, y_train)
    time_base_model = time.time() - start_time

    ensemble_model.set_params(n_estimators=100)
    start_time = time.time()
    ensemble_model.fit(X_train, y_train)
    time_ensemble = time.time() - start_time

    print(f"Training Time for Base Model: {time_base_model:.4f} seconds")
    print(f"Training Time for Ensemble Model: {time_ensemble:.4f} seconds")


def metrics(X_train, y_train, X_test, y_test):
    print("\nMetrics:")
    base_model = DecisionTreeRegressor(max_depth=3, random_state=42)
    base_model.fit(X_train, y_train)
    ensemble_model = AdaBoostRegressor(random_state=42)
    ensemble_model.fit(X_train, y_train)

    y_test_pred_base = base_model.predict(X_test)
    r2_base = r2_score(y_test, y_test_pred_base)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_test_pred_base))
    mape_base = mean_absolute_percentage_error(y_test, y_test_pred_base)

    y_test_pred_ensemble = ensemble_model.predict(X_test)
    r2_ensemble = r2_score(y_test, y_test_pred_ensemble)
    rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_test_pred_ensemble))
    mape_ensemble = mean_absolute_percentage_error(y_test, y_test_pred_ensemble)

    print("Performance Comparison:")
    print(
        f"Base Model - R2: {r2_base:.4f}, RMSE: {rmse_base:.4f}, MAPE: {mape_base:.4f}")
    print(
        f"Ensemble - R2: {r2_ensemble:.4f}, RMSE: {rmse_ensemble:.4f}, MAPE: {mape_ensemble:.4f}")


if __name__ == "__main__":
    # Task 1
    X, y = visual_dataset()

    # Task 2
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    # Task 3, 4
    build_ensemble_models(X_val, y_val, X_test, y_test)

    # Task 5
    bias_and_variance(X_train, y_train, X_test, y_test)

    # Task 6
    timing(X_train, y_train)

    # Task 7
    metrics(X_train, y_train, X_test, y_test)
