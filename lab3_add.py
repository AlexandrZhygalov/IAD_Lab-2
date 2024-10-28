from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, load_iris
from sklearn.model_selection import train_test_split

# Параметри даних для make_circles
X, y = make_circles(n_samples=500, factor=0.1, noise=0.1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Початкові параметри нейронної мережі
max_neurons = 50
tolerance = 0.001  # Порогова зміна для зупинки
neurons_range = range(1, max_neurons + 1)

train_errors, val_errors = [], []

# Модель для make_circles
for neurons in neurons_range:
    model = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Прогнози
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Оцінка метрик
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    train_errors.append(train_mse)
    val_errors.append(val_mse)

    # Перевірка на стабілізацію помилки
    if neurons > 1 and abs(val_errors[-2] - val_mse) < tolerance:
        print(f"Оптимальна кількість нейронів для make_circles: {neurons}")
        break

# Візуалізація
plt.plot(neurons_range[:len(val_errors)], train_errors, label="Train MSE (make_circles)")
plt.plot(neurons_range[:len(val_errors)], val_errors, label="Validation MSE (make_circles)")
plt.xlabel("Кількість нейронів")
plt.ylabel("MSE")
plt.legend()
plt.title("Динаміка помилки в залежності від кількості нейронів (make_circles)")
plt.show()

# --- Аналогічні операції для Iris dataset ---
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_iris, X_val_iris, y_train_iris, y_val_iris = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

train_errors_iris, val_errors_iris = [], []

# Модель для Iris
for neurons in neurons_range:
    model_iris = MLPRegressor(hidden_layer_sizes=(neurons,), max_iter=1000, random_state=42)
    model_iris.fit(X_train_iris, y_train_iris)

    # Прогнози
    y_train_pred_iris = model_iris.predict(X_train_iris)
    y_val_pred_iris = model_iris.predict(X_val_iris)

    # Оцінка метрик
    train_mse_iris = mean_squared_error(y_train_iris, y_train_pred_iris)
    val_mse_iris = mean_squared_error(y_val_iris, y_val_pred_iris)

    train_errors_iris.append(train_mse_iris)
    val_errors_iris.append(val_mse_iris)

    # Перевірка на стабілізацію помилки
    if neurons > 1 and abs(val_errors_iris[-2] - val_mse_iris) < tolerance:
        print(f"Оптимальна кількість нейронів для Iris: {neurons}")
        break

# Візуалізація для Iris
plt.plot(neurons_range[:len(val_errors_iris)], train_errors_iris, label="Train MSE (Iris)")
plt.plot(neurons_range[:len(val_errors_iris)], val_errors_iris, label="Validation MSE (Iris)")
plt.xlabel("Кількість нейронів")
plt.ylabel("MSE")
plt.legend()
plt.title("Динаміка помилки в залежності від кількості нейронів (Iris)")
plt.show()
