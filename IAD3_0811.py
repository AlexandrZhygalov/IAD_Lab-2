import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from scipy import stats


# 1. Представити початковi данi графiчно
def create_and_show_dataset():
    # Датасет 'Кільця'
    X_circles, Y_circles = make_circles(n_samples=500, factor=0.1, noise=0.1)
    plt.figure(figsize=(6, 6))
    plt.scatter(X_circles[:, 0], X_circles[:, 1], c=Y_circles, cmap='viridis')
    plt.title("Набір 'Кільця'")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Датасет 'Ірис'
    X_iris, Y_iris = load_iris(return_X_y=True)
    sns.pairplot(sns.load_dataset("iris"), hue="species")
    plt.show()
    return X_circles, Y_circles, X_iris, Y_iris


X_circles, Y_circles, X_iris, Y_iris = create_and_show_dataset()


# 2. Розбити данi на навчальний та валiдацiйний набори
X_train_circles, X_val_circles, y_train_circles, y_val_circles = train_test_split(
    X_circles, Y_circles, test_size=0.3, random_state=42)
X_train_iris, X_val_iris, y_train_iris, y_val_iris = train_test_split(
    X_iris, Y_iris, test_size=0.3, random_state=42)


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
        model = MLPRegressor(hidden_layer_sizes=(n_neurons,), max_iter=5000, random_state=42, solver='lbfgs')

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

    print("\nНе вдалося досягти цільових метрик.")
    return None, None  # Якщо цільових значень не було досягнуто


# Виклик функції для наборів даних Circles та Iris
print("Оптимізація для Circles:")
optimal_circle_model, optimal_circle_neurons = find_optimal_neurons(X_train_circles, y_train_circles, X_val_circles,
                                                                    y_val_circles)

print("\nОптимізація для Iris:")
optimal_iris_model, optimal_iris_neurons = find_optimal_neurons(X_train_iris, y_train_iris, X_val_iris, y_val_iris)


# 4. Побудова та навчання дерева рішень для кожного набору даних
circle_tree_regressor = DecisionTreeRegressor(random_state=42)
circle_tree_regressor.fit(X_train_circles, y_train_circles)
y_pred_circles_reg = circle_tree_regressor.predict(X_val_circles)

iris_tree_regressor = DecisionTreeRegressor(random_state=42)
iris_tree_regressor.fit(X_train_iris, y_train_iris)
y_pred_iris_reg = iris_tree_regressor.predict(X_val_iris)


plt.figure(figsize=(14, 10))  # збільшення розміру
plot_tree(
    circle_tree_regressor,
    filled=True,
    feature_names=["Feature 1", "Feature 2"],
    precision=2,  # зменшуємо кількість знаків після коми
    impurity=False,  # не відображаємо impurity, щоб зменшити кількість інформації
    rounded=True,  # округлі вузли для кращої візуалізації
    fontsize=10,  # налаштування розміру шрифту
)
plt.title("Дерево рішень для регресії (Circles)", fontsize=16)
plt.show()

# Візуалізація дерева для набору "Iris"
plt.figure(figsize=(14, 10))
plot_tree(
    iris_tree_regressor,
    filled=True,
    feature_names=load_iris().feature_names,
    precision=2,
    impurity=False,
    rounded=True,
    fontsize=10,
)
plt.title("Дерево рішень для регресії (Iris)", fontsize=16)
plt.show()


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


# 7. Розрахунок апостеріорних ймовірностей для моделей
# Функція для розрахунку апостеріорних ймовірностей
circle_val_pred = optimal_circle_model.predict(X_val_circles)
circle_residuals = y_val_circles - circle_val_pred

iris_val_pred = optimal_iris_model.predict(X_val_iris)
iris_residuals = y_val_iris - iris_val_pred

# Функція для розрахунку інтервалу довіри
def confidence_interval(predictions, residuals, confidence=0.95):
    n = len(predictions)
    mean_prediction = np.mean(predictions)
    std_error = np.std(residuals, ddof=1) / np.sqrt(n)
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)

    margin_of_error = t_critical * std_error
    lower_bound = mean_prediction - margin_of_error
    upper_bound = mean_prediction + margin_of_error

    return lower_bound, upper_bound

# Розрахунок інтервалу довіри для моделей
circle_ci = confidence_interval(circle_val_pred, circle_residuals)
iris_ci = confidence_interval(iris_val_pred, iris_residuals)

print("\nІнтервали довіри:")
print(f"Інтервал довіри для моделі 'Circles' (95%): ({round(float(circle_ci[0]), 2)}, {round(float(circle_ci[1]), 2)})")
print(f"Інтервал довіри для моделі 'Iris' (95%): ({round(float(iris_ci[0]), 2)}, {round(float(iris_ci[1]), 2)})")


# 10 завдання
# Функція для обчислення метрик
print("\nМетрики:")
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
    print(f"R^2 (Train): {r2_train:.4f}, R^2 (Validation): {r2_val:.4f}")
    print(f"RMSE (Train): {rmse_train:.4f}, RMSE (Validation): {rmse_val:.4f}")
    print(f"MAE (Train): {mae_train:.4f}, MAE (Validation): {mae_val:.4f}")
    print(f"MAPE (Train): {mape_train:.2f}%, MAPE (Validation): {mape_val:.2f}%")

    # Повернення значень для візуалізації
    return {
        'R^2': [round(r2_train, 4), round(r2_val, 4)],
        'RMSE': [round(rmse_train, 4), round(rmse_val, 4)],
        'MAE': [round(mae_train, 4), round(mae_val, 4)],
        'MAPE, %': [round(mape_train, 4), round(mape_val, 4)]
    }


def plot_metrics_table(metrics, title):
    df = pd.DataFrame(metrics, index=["Train set", "Validation set"]).applymap("{:.4f}".format)

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    plt.suptitle(title, fontsize=10, ha='center', y=0.8)
    plt.subplots_adjust(left=0.2)
    plt.show()


# Розрахунок метрик для 'Circles' та 'Iris'
print("Метрики для 'Circles'")
metrics_circles = calculate_metrics(optimal_circle_model, X_train_circles, y_train_circles, X_val_circles,
                                    y_val_circles)
plot_metrics_table(metrics_circles, "Метрики для 'Circles'")

print("\nМетрики для 'Iris'")
metrics_iris = calculate_metrics(optimal_iris_model, X_train_iris, y_train_iris, X_val_iris, y_val_iris)
plot_metrics_table(metrics_iris, "Метрики для 'Iris'")


# 11 завдання
# Параметри для grid search
print("\nGrid search:")
param_grid = {'hidden_layer_sizes': [1, 10, 20, 30, 40, 100],
'activation': ['tanh', 'relu'],
'random_state': [0],
'max_iter': [15000],
'solver': ['lbfgs', 'sgd', 'adam']}

# Ініціалізація моделі
mlp = MLPRegressor(max_iter=15000, random_state=42)

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


def create_best_param_models():
    best_param_circle_reg = MLPRegressor(activation=best_params_circles['activation'],
                                         random_state=best_params_circles['random_state'],
                                         hidden_layer_sizes=best_params_circles['hidden_layer_sizes'],
                                         max_iter=best_params_circles['max_iter'],
                                         solver=best_params_circles['solver'])

    best_param_iris_reg = MLPRegressor(activation=best_params_iris['activation'],
                                       random_state=best_params_iris['random_state'],
                                       hidden_layer_sizes=best_params_iris['hidden_layer_sizes'],
                                       max_iter=best_params_iris['max_iter'],
                                       solver=best_params_iris['solver'])

    best_param_circle_reg.fit(X_train_circles, y_train_circles)
    best_param_iris_reg.fit(X_train_iris, y_train_iris)

    print("Метрики найкращої моделі для 'Circles'")
    best_params_metric_circles = calculate_metrics(best_param_circle_reg,
                                                   X_train_circles,
                                                   y_train_circles,
                                                   X_val_circles,
                                                   y_val_circles)

    print("\nМетрики найкращої моделі для 'Iris'")
    best_params_metric_iris = calculate_metrics(best_param_iris_reg,
                                                   X_train_iris,
                                                   y_train_iris,
                                                   X_val_iris,
                                                   y_val_iris)

    return best_params_metric_circles, best_params_metric_iris


print("\nМетрики моделей з найкращими параметрами:")
best_params_metric_circles, best_params_metric_iris = create_best_param_models()
plot_metrics_table(best_params_metric_circles, "Метрики моделі з найкращими параметрами для 'Circles'")
plot_metrics_table(best_params_metric_iris, "Метрики моделі з найкращими параметрами для 'Iris'")


# 12 завдання
def compare_metrics(metrics, best_params_metric, title):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)

    for i, (metric, values_1) in enumerate(metrics.items()):
        values_2 = best_params_metric[metric]

        train_values = [values_1[0], values_2[0]]
        val_values = [values_1[1], values_2[1]]

        ax = axs[i // 2, i % 2]

        x = np.arange(2)
        width = 0.3

        ax.bar(x - width / 2, train_values, width=width, label='Тренувальний набір', color='green', alpha=0.5)
        ax.bar(x + width / 2, val_values, width=width, label='Валідаційна', color='blue', alpha=0.5)

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(['Модель звичайна', 'Модель з найкращими параметрами'])
        ax.set_ylabel('Значення')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


compare_metrics(metrics_circles, best_params_metric_circles, "Порівняння метрик для двох моделей для набору 'Circle'")
compare_metrics(metrics_iris, best_params_metric_iris, "Порівняння метрик для двох моделей для набору 'Iris'")


# 13 завдання
print("\nОцінки наборів:")
def evaluate_on_different_sample_sizes_regression(X, y, sample_sizes, model_class, **kwargs):
    results = []
    for size in sample_sizes[:-1]:
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=size, random_state=42)
        model = model_class(random_state=42, **kwargs)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        print(f"Sample Size: {size:.0%} - Validation MSE: {mse:.4f}, R²: {r2:.4f}")
        results.append((size, mse, r2))
    return results

sample_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.99, 1]

# Функція для візуалізації результатів
def plot_metrics(results, dataset_name):
    # Перетворюємо результати у DataFrame
    results_df = pd.DataFrame(results[:-1], columns=['Sample Size', 'MSE', 'R²'])

    # Створення графіків для кожної метрики
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MSE графік
    axes[0].plot(results_df['Sample Size'], results_df['MSE'], marker='o', color='b', label='MSE')
    axes[0].set_title(f"{dataset_name} - MSE vs. Sample Size")
    axes[0].set_xlabel("Sample Size (fraction)")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True)

    # R² графік
    axes[1].plot(results_df['Sample Size'], results_df['R²'], marker='o', color='r', label='R²')
    axes[1].set_title(f"{dataset_name} - R² vs. Sample Size")
    axes[1].set_xlabel("Sample Size (fraction)")
    axes[1].set_ylabel("R²")
    axes[1].grid(True)

    # Відображення графіків
    plt.tight_layout()
    plt.show()

# Приклад виклику функції
print("Evaluating Circles Dataset:")
results_circles = evaluate_on_different_sample_sizes_regression(X_circles, Y_circles, sample_sizes,
                                                                MLPRegressor, max_iter=5000)
plot_metrics(results_circles, "Circles Dataset")

print("\nEvaluating Iris Dataset:")
results_iris = evaluate_on_different_sample_sizes_regression(X_iris, Y_iris, sample_sizes,
                                                             MLPRegressor, max_iter=5000)
plot_metrics(results_iris, "Iris Dataset")
