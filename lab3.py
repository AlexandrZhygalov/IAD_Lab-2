import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.decomposition import PCA
from sklearn.datasets import make_circles, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score, mean_squared_error, r2_score, mean_absolute_error)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


# 1 завдання - Представити початковi данi графiчно.
X, y = make_circles(n_samples=500, factor=0.1, noise=0.1)
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title("Набір 'Кільця'")
plt.show()

X_iris, y_iris = load_iris(return_X_y=True)
sns.pairplot(sns.load_dataset("iris"), hue="species")
plt.show()


# 2 завдання - Розбити данi на навчальний та валiдацiйний набори.
X_train_circles, X_val_circles, y_train_circles, y_val_circles = train_test_split(
    X, y, test_size=0.3, random_state=42)

X_train_iris, X_val_iris, y_train_iris, y_val_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42)


# 3 завдання.
iris_reg = MLPRegressor(random_state=0, hidden_layer_sizes=1, max_iter=5000)
circle_reg = MLPRegressor(random_state=0, hidden_layer_sizes=1, max_iter=5000)

iris_reg.fit(X_train_iris, y_train_iris)
circle_reg.fit(X_train_circles, y_train_circles)


# 4 завдання - Представити моделi графiчно (наприклад вивести частину дерева рiшень, побудувати лiнiю регресiї тощо).
# Функція для візуалізації моделей логістичної регресії кільця
def plot_logistic_regression_comparison(X, y, models, titles):
    for model, title in zip(models, titles):
        plt.figure(figsize=(8, 6))

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis')
        plt.title(title)
        plt.xlabel('Перша ознака')
        plt.ylabel('Друга ознака')

        plt.tight_layout()
        plt.show()


models_circles = [circle_reg]

titles_circles = ["Набір даних кільця"]

plot_logistic_regression_comparison(X_train_circles, y_train_circles, models_circles, titles_circles)


def plot_iris_models_comparison(models, titles):
    pca = PCA(n_components=2)
    X_iris_pca = pca.fit_transform(X_train_iris)

    for model, title in zip(models, titles):
        plt.figure(figsize=(8, 6))

        x_min, x_max = X_iris_pca[:, 0].min() - 1, X_iris_pca[:, 0].max() + 1
        y_min, y_max = X_iris_pca[:, 1].min() - 1, X_iris_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        grid_points = np.c_[xx.ravel(), yy.ravel()]

        Z_mesh = model.predict(pca.inverse_transform(grid_points))
        Z_mesh = Z_mesh.reshape(xx.shape)

        plt.contourf(xx, yy, Z_mesh, alpha=0.3, cmap='viridis')

        scatter = plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_train_iris, edgecolors='k', cmap='viridis')
        plt.title(title)
        plt.xlabel('Перша головна складова')
        plt.ylabel('Друга головна складова')

        plt.tight_layout()
        plt.show()


models_iris = [iris_reg]

titles_iris = ["Набір даних Iris"]

plot_iris_models_comparison(models_iris, titles_iris)


# 5 завдання - Виконати прогнози на основi побудованих моделей.
y_val_pred_circles = circle_reg.predict(X_val_circles)
y_val_pred_iris = iris_reg.predict(X_val_iris)

y_train_pred_circles = circle_reg.predict(X_train_circles)
y_train_pred_iris = iris_reg.predict(X_train_iris)

y_train_pred_circles_class = (y_train_pred_circles > 0.5).astype(int)
y_val_pred_circles_class = (y_val_pred_circles > 0.5).astype(int)

y_train_pred_iris_class = (y_train_pred_iris > 0.5).astype(int)
y_val_pred_iris_class = (y_val_pred_iris > 0.5).astype(int)


# 6 завдання - Для кожної з моделей оцiнити, чи має мiсце перенавчання.
print("\nЗавдання 6")
train_accuracy_circle = accuracy_score(y_train_pred_circles_class, y_train_pred_circles_class)
val_accuracy_circle = accuracy_score(y_val_pred_circles_class, y_val_pred_circles_class)

train_accuracy_iris_multi_no_reg = accuracy_score(y_train_pred_iris_class, y_train_pred_iris_class)
val_accuracy_iris_simple_no_reg = accuracy_score(y_val_pred_iris_class, y_val_pred_iris_class)


print(f"\nТочність для навчального набору Сircle: {train_accuracy_circle:.5f}")
print(f"Точність для валідаційного набору Сircle: {val_accuracy_circle:.5f}")
print("\nЗвіт по метриках для набору Сircle, simple, reg:")
print(classification_report(y_val_circles, y_val_pred_circles_class))

print(f"\nТочність для навчального набору Iris: {train_accuracy_circle:.5f}")
print(f"Точність для валідаційного набору Iris: {val_accuracy_circle:.5f}")
print("\nЗвіт по метриках для набору Iris:")
print(classification_report(y_val_circles, y_val_pred_circles_class))


# 7 завдання - Розрахувати додатковi результати моделей, наприклад, апостерiорнi iмовiрностi або iншi (згiдно з варiантом).
print("\nЗавдання 7")
# Оцінка невизначеності для make_circles
residuals_circles = y_val_circles - y_val_pred_circles
residual_std_circles = np.std(residuals_circles)
posterior_probabilities_circles = 1 - np.exp(-((y_val_circles - y_val_pred_circles) ** 2) / (2 * residual_std_circles ** 2))

mean_posterior_circles = np.mean(posterior_probabilities_circles)
print("Середнє апостеріорне ймовірність для Circles:", mean_posterior_circles)

# Оцінка невизначеності для Iris
residuals_iris = y_val_iris - y_val_pred_iris
residual_std_iris = np.std(residuals_iris)
posterior_probabilities_iris = 1 - np.exp(-((y_val_iris - y_val_pred_iris) ** 2) / (2 * residual_std_iris ** 2))

mean_posterior_iris = np.mean(posterior_probabilities_iris)
print("Середнє апостеріорне ймовірність для Iris:", mean_posterior_iris)



# 10 завдання - В задачах регресiї розрахувати для кожної моделi наступнi критерiї якостi, окремо на навчальнiй та валiдацiйнiй множинах:
# • коефiцiєнт детермiнацiї R2
# • помилки RMSE, MAE та MAPE.
print("\nЗавдання 10")
def evaluate_regression_model(y_true, y_pred, model_name=""):
    # R²
    r2 = r2_score(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mae = mean_absolute_error(y_true, y_pred)

    non_zero_indices = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

    print(f"\nОцінка моделі {model_name}:")
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"MAPE: {mape:.3f}%")


def evaluate_models(models):
    for model_name, (model, X_train, y_train, X_val, y_val) in models.items():
        y_train_pred = model.predict(X_train)
        print(f"\nОцінка на навчальній множині для {model_name}:")
        evaluate_regression_model(y_train, y_train_pred, model_name)

        y_val_pred = model.predict(X_val)
        print(f"\nОцінка на валідаційній множині для {model_name}:")
        evaluate_regression_model(y_val, y_val_pred, model_name)


models = {
    "Набір даних Circle": (
        circle_reg, X_train_circles, y_train_circles, X_val_circles, y_val_circles),
    "Набір даних Iris": (
        iris_reg, X_train_iris, y_train_iris, X_val_iris, y_val_iris)
}

evaluate_models(models)


# 11 завдання - Спробувати виконати решiтчастий пошук (grid search) для пiдбору гiперпараметрiв моделей.
print("\nЗавдання 11")
warnings.filterwarnings("ignore")
param_grid = {'hidden_layer_sizes': [1, 10, 20, 30, 40, 100],
              'activation': ['tanh', 'relu'],
              'random_state': [0],
              'max_iter': [5000],
              'solver': ['lbfgs', 'sgd', 'adam']}


grid_search_circle = GridSearchCV(MLPRegressor(), param_grid, cv=5, scoring='r2')
grid_search_circle.fit(X_train_circles, y_train_circles)
print("Найкращі параметри для логістичної регресії (Circle): ", grid_search_circle.best_params_)
print("Найкраща точність: ", grid_search_circle.best_score_)

grid_search_iris = GridSearchCV(MLPRegressor(), param_grid, cv=5, scoring='r2')
grid_search_iris.fit(X_train_iris, y_train_iris)
print("Найкращі параметри для логістичної регресії (Iris): ", grid_search_iris.best_params_)
print("Найкраща точність: ", grid_search_iris.best_score_)


# 13 завдання - Навчити моделi на пiдмножинах навчальних даних. Оцiнити, наскiльки
# розмiр навчальної множини впливає на якiсть моделi.
print("\nЗавдання 13")
def evaluate_on_different_sample_sizes(X, y, sample_sizes, model_class, **kwargs):
    results = []
    for size in sample_sizes:
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=size, random_state=42)
        model = model_class(**kwargs)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_val_pred_class = (y_val_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_val_pred_class)
        results.append((size, accuracy))

        print(f"Sample Size: {size:.0%} - Accuracy: {accuracy:.4f}")
        print(classification_report(y_val, y_val_pred_class))

    return results


sample_sizes = [0.1, 0.2, 0.5, 0.99]  # 10%, 20%, 50%, 99%


print("Evaluating Circles Dataset (simple):")
evaluate_on_different_sample_sizes(X, y, sample_sizes, MLPRegressor)


print("\nEvaluating Iris Dataset:")
evaluate_on_different_sample_sizes(X_iris, y_iris, sample_sizes, MLPRegressor)
