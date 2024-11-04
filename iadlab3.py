import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 1. Представити початковi данi графiчно
# Датасет 'Кільця'
X_circles, y_circles = make_circles(n_samples=500, factor=0.1, noise=0.1)
plt.figure(figsize=(6, 6))
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis')
plt.title("Набір 'Кільця'")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Датасет 'Ірис'
X_iris, y_iris = load_iris(return_X_y=True)
sns.pairplot(sns.load_dataset("iris"), hue="species")
plt.show()

# 2. Розбити данi на навчальний та валiдацiйний набори
X_train_circles, X_val_circles, y_train_circles, y_val_circles = train_test_split(
    X_circles, y_circles, test_size=0.3, random_state=42)
X_train_iris, X_val_iris, y_train_iris, y_val_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42)

# 3. Побудувати на навчальному наборi даних моделi регресії
circle_reg = MLPRegressor(random_state=0, hidden_layer_sizes=1, max_iter=5000)
iris_reg = MLPRegressor(random_state=0, hidden_layer_sizes=1, max_iter=5000)

circle_reg.fit(X_train_circles, y_train_circles)
iris_reg.fit(X_train_iris, y_train_iris)

#збільшення нейронів

# Граничні значення для визначення задовільної моделі
TARGET_MSE = 0.1  # Цільове значення MSE (можна змінити за потреби)
TARGET_R2 = 0.9  # Цільове значення R^2 (можна змінити за потреби)
MAX_NEURONS = 20  # Максимальна кількість нейронів для перевірки

# Функція для динамічного збільшення кількості нейронів
def find_optimal_neurons(X_train, y_train, X_val, y_val):
    for n_neurons in range(1, MAX_NEURONS + 1):
        # Визначення одношарової моделі з n_neurons нейронами
        model = MLPRegressor(hidden_layer_sizes=(n_neurons,), max_iter=5000, random_state=0, solver='lbfgs')

        # Навчання моделі
        model.fit(X_train, y_train)

        # Прогноз на валідаційному наборі
        y_val_pred = model.predict(X_val)

        # Обчислення MSE та R^2
        mse = mean_squared_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)

        print(f"Кількість нейронів: {n_neurons}, MSE: {mse:.4f}, R^2: {r2:.4f}")

        # Перевірка, чи досягла модель цільових значень
        if mse <= TARGET_MSE and r2 >= TARGET_R2:
            print(f"\nДосягнуто цільових метрик з {n_neurons} нейронами.")
            return model, n_neurons  # Повернення оптимальної моделі і кількості нейронів

    print("\nНе вдалося досягти цільових метрик. Розгляньте збільшення MAX_NEURONS або зміну цільових значень.")
    return None, None  # Якщо цільових значень не було досягнуто


# Виклик функції для наборів даних Circles та Iris
print("Оптимізація для Circles:")
optimal_circle_model, optimal_circle_neurons = find_optimal_neurons(X_train_circles, y_train_circles, X_val_circles,
                                                                    y_val_circles)

print("\nОптимізація для Iris:")
optimal_iris_model, optimal_iris_neurons = find_optimal_neurons(X_train_iris, y_train_iris, X_val_iris, y_val_iris)
#
# #Завдання 4 доробити !!!
#
# 5. Виконати прогнози на основі побудованих оптимальних моделей
y_pred_circles = optimal_circle_model.predict(X_val_circles)
y_pred_iris = optimal_iris_model.predict(X_val_iris)

# 6. Оцінка перенавчання для оптимальних моделей
# Розрахунок MSE та R² для навчальних і валідаційних наборів
mse_train_circles = mean_squared_error(y_train_circles, optimal_circle_model.predict(X_train_circles))
mse_val_circles = mean_squared_error(y_val_circles, y_pred_circles)
r2_train_circles = r2_score(y_train_circles, optimal_circle_model.predict(X_train_circles))
r2_val_circles = r2_score(y_val_circles, y_pred_circles)

mse_train_iris = mean_squared_error(y_train_iris, optimal_iris_model.predict(X_train_iris))
mse_val_iris = mean_squared_error(y_val_iris, y_pred_iris)
r2_train_iris = r2_score(y_train_iris, optimal_iris_model.predict(X_train_iris))
r2_val_iris = r2_score(y_val_iris, y_pred_iris)

print("\nОптимальна Circle Regression:")
print(f"  Training MSE: {mse_train_circles:.4f}, Validation MSE: {mse_val_circles:.4f}")
print(f"  Training R²: {r2_train_circles:.4f}, Validation R²: {r2_val_circles:.4f}")

print("\nОптимальна Iris Regression:")
print(f"  Training MSE: {mse_train_iris:.4f}, Validation MSE: {mse_val_iris:.4f}")
print(f"  Training R²: {r2_train_iris:.4f}, Validation R²: {r2_val_iris:.4f}")

#10 завдання
# Функція для обчислення метрик
def calculate_metrics(model, X_train, y_train, X_val, y_val):
    # Прогнози на тренувальній та валідаційній множинах
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Коефіцієнт детермінації R2
    r2_train = r2_score(y_train, y_train_pred)
    r2_val = r2_score(y_val, y_val_pred)

    # RMSE
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)

    # MAE
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)

    # MAPE з уникненням ділення на нуль
    non_zero_train = y_train != 0
    non_zero_val = y_val != 0
    mape_train = np.mean(
        np.abs((y_train[non_zero_train] - y_train_pred[non_zero_train]) / y_train[non_zero_train])) * 100
    mape_val = np.mean(np.abs((y_val[non_zero_val] - y_val_pred[non_zero_val]) / y_val[non_zero_val])) * 100

    # Виведення результатів
    print(f"R2 (Train): {r2_train:.4f}, R2 (Validation): {r2_val:.4f}")
    print(f"RMSE (Train): {rmse_train:.4f}, RMSE (Validation): {rmse_val:.4f}")
    print(f"MAE (Train): {mae_train:.4f}, MAE (Validation): {mae_val:.4f}")
    print(f"MAPE (Train): {mape_train:.2f}%, MAPE (Validation): {mape_val:.2f}%")

    # Повернення значень для візуалізації
    return {
        'R2': [r2_train, r2_val],
        'RMSE': [rmse_train, rmse_val],
        'MAE': [mae_train, mae_val],
        'MAPE': [mape_train, mape_val]
    }


# Розрахунок метрик для 'Circles' та 'Iris'
print("Метрики для 'Circles'")
metrics_circles = calculate_metrics(optimal_circle_model, X_train_circles, y_train_circles, X_val_circles,
                                    y_val_circles)

print("\nМетрики для 'Iris'")
metrics_iris = calculate_metrics(optimal_iris_model, X_train_iris, y_train_iris, X_val_iris, y_val_iris)

#11 завдання
from sklearn.model_selection import GridSearchCV

# Параметри для grid search

param_grid = {'hidden_layer_sizes': [1, 10, 20, 30, 40, 100],
'activation': ['tanh', 'relu'],
'random_state': [0],
'max_iter': [5000],
'solver': ['lbfgs', 'sgd', 'adam']}

# Ініціалізація моделі
mlp = MLPRegressor(max_iter=5000, random_state=42)

# Ініціалізація grid search з перехресною перевіркою
grid_search_circles = GridSearchCV(estimator=mlp, param_grid=param_grid,
                                     scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search_iris = GridSearchCV(estimator=mlp, param_grid=param_grid,
                                  scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

# Запуск grid search для наборів даних
grid_search_circles.fit(X_train_circles, y_train_circles)
grid_search_iris.fit(X_train_iris, y_train_iris)

# Отримання найкращих параметрів
best_params_circles = grid_search_circles.best_params_
best_params_iris = grid_search_iris.best_params_

print("Найкращі параметри для Circles:", best_params_circles)
print("Найкращі параметри для Iris:", best_params_iris)

#12 завдання

# Оцінка якості моделей
models = {
    "Optimal Circles": optimal_circle_model,
    "Optimal Iris": optimal_iris_model
}

# Функція для обчислення метрик і виведення результатів
def evaluate_models(models, X_train, y_train, X_val, y_val):
    for name, model in models.items():
        # Прогнози на навчальних і валідаційних наборах
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Обчислення MSE та R²
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_val = mean_squared_error(y_val, y_val_pred)
        r2_train = r2_score(y_train, y_train_pred)
        r2_val = r2_score(y_val, y_val_pred)

        print(f"{name}:")
        print(f"  Training MSE: {mse_train:.4f}, Validation MSE: {mse_val:.4f}")
        print(f"  Training R²: {r2_train:.4f}, Validation R²: {r2_val:.4f}")

# Виклик функції для оцінки моделей
evaluate_models(models, X_train_circles, y_train_circles, X_val_circles, y_val_circles)
evaluate_models(models, X_train_iris, y_train_iris, X_val_iris, y_val_iris)



