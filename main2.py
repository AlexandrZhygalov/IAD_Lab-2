import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.decomposition import PCA
from sklearn.datasets import make_circles, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score, mean_squared_error, r2_score, mean_absolute_error)





# 1 завдання - Представити початковi данi графiчно.
"""
Набір даних "кільця":

На графіку буде зображено двокласовий набір точок, створений функцією make_circles. Він матиме форму концентричних кіл 
або кілець, де один клас точок розташовується у внутрішньому кільці, а інший — у зовнішньому. Розподіл точок дещо 
зашумлений, що відображає параметр noise=0.1, надаючи набору більш реалістичний вигляд.

Набір даних Iris:

Ви побачите множину діаграм розсіювання (pairplot), де різні пари ознак (довжина та ширина чашолистків і пелюсток) 
порівнюються між собою для трьох класів (видів) квітів Iris. Це дозволяє швидко оцінити, як розділяються класи в цьому 
просторі ознак і чи є якісь класи, які перекриваються за певними характеристиками.
"""
# Згенерований набір "кільця"
X, y = make_circles(n_samples=500, factor=0.1, noise=0.1)
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title("Набір 'Кільця'")
plt.show()

# Набір Iris
X_iris, y_iris = load_iris(return_X_y=True)
sns.pairplot(sns.load_dataset("iris"), hue="species")
plt.show()


# 2 завдання - Розбити данi на навчальний та валiдацiйний набори.
"""
Виклик train_test_split:

Ми передаємо функції train_test_split вхідні дані (X, ознаки) і мітки класів (y).
Зазначаємо параметр test_size=0.3, що означає: 30% даних будуть відведені для валідаційного набору, 
а решта 70% — для навчального.
Параметр random_state=42 фіксує порядок розбиття, забезпечуючи повторюваність результатів при кожному запуску.
Якщо не встановлювати random_state, результат може змінюватися при кожному запуску, 
оскільки випадковий процес буде ініціалізований випадковим значенням.

Повернуті значення:

train_test_split повертає чотири набори: X_train і y_train (для навчання), X_val і y_val (для валідації).
"""

# Згенерований набір "кільця"
X_train_circles, X_val_circles, y_train_circles, y_val_circles = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Набір Iris
X_train_iris, X_val_iris, y_train_iris, y_val_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42)


# 3 завдання - Побудувати на навчальному наборi даних моделi класифiкацiї або регресiї заданi згiдно з варiантом.

"""
Логістична регресія — це метод класифікації, що прогнозує ймовірність належності до певного класу. Ключова мета 
логістичної регресії — розділити дані на два або більше класів, використовуючи математичну функцію, яка 
перетворює лінійну комбінацію ознак на ймовірність приналежності до певного класу.

Деталі пункту:
Проста логістична регресія:
Створюємо об’єкт LogisticRegression() для простої логістичної регресії без додаткових параметрів.
Навчаємо модель на навчальних даних.
Це буде стандартна модель, яка класифікує дані на основі лінійного поділу.

Поліноміальна (multinomial) логістична регресія:
Створюємо модель з параметрами multi_class='multinomial' і solver='lbfgs', що дозволяє моделі обробляти випадки з 
більш ніж двома класами. Поліноміальна логістична регресія (також відома як багатокласова) використовується, коли 
класів більше двох (наприклад, у наборі даних Iris, де є три класи). Вона прогнозує ймовірність приналежності до 
кожного з класів одночасно, що дозволяє їй працювати в багатокласових задачах класифікації.

Моделі з регуляризацією та без:

Регуляризація — це метод, що допомагає запобігти перенавчанню, додаючи "штраф" за складні моделі.
Ми створюємо варіанти моделей з регуляризацією та без неї, що дозволить оцінити вплив регуляризації на 
точність та узагальнювальну здатність моделей.

Навчання виконується для кожної моделі на навчальних даних (X_train і y_train).
Моделі намагаються знайти оптимальні ваги для ознак, щоб мінімізувати помилку класифікації.
"""

# Проста логістична регресія
simple_log_reg_circles = LogisticRegression()
simple_log_reg_iris = LogisticRegression()

# Поліноміальна логістична регресія
multi_log_reg_circles = LogisticRegression(multi_class='multinomial', solver='lbfgs')
multi_log_reg_iris = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Навчання моделей
simple_log_reg_circles.fit(X_train_circles, y_train_circles)
simple_log_reg_iris.fit(X_train_iris, y_train_iris)
multi_log_reg_circles.fit(X_train_circles, y_train_circles)
multi_log_reg_iris.fit(X_train_iris, y_train_iris)


# Проста логістична регресія без регуляції
simple_log_no_reg_circles = LogisticRegression(C=0.1)
simple_log_no_reg_iris = LogisticRegression(C=0.1)

# Поліноміальна логістична регресія без регуляції
multi_log_no_reg_circles = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=0.1)
multi_log_no_reg_iris = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=0.1)

# Навчання моделей без регуляції
simple_log_no_reg_circles.fit(X_train_circles, y_train_circles)
simple_log_no_reg_iris.fit(X_train_iris, y_train_iris)
multi_log_no_reg_circles.fit(X_train_circles, y_train_circles)
multi_log_no_reg_iris.fit(X_train_iris, y_train_iris)


# 4 завдання - Представити моделi графiчно (наприклад вивести частину дерева рiшень, побудувати лiнiю регресiї тощо).
# Функція для візуалізації моделей логістичної регресії кільця
def plot_logistic_regression_comparison(X, y, models, titles):
    for model, title in zip(models, titles):
        plt.figure(figsize=(8, 6))

        # Визначити межі
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        # Прогнозування для кожної точки на сітці
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Відобразити контур
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

        # Відобразити дані
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis')
        plt.title(title)
        plt.xlabel('Перша ознака')
        plt.ylabel('Друга ознака')

        plt.tight_layout()
        plt.show()


# Графічне представлення моделей для набору "кільця"
models_circles = [
    simple_log_reg_circles,
    multi_log_reg_circles,
    simple_log_no_reg_circles,
    multi_log_no_reg_circles
]

titles_circles = [
    "Проста логістична регресія (Регуляризація)",
    "Поліноміальна логістична регресія (Регуляризація)",
    "Проста логістична регресія (Без регуляризації)",
    "Поліноміальна логістична регресія (Без регуляризації)"
]

plot_logistic_regression_comparison(X_train_circles, y_train_circles, models_circles, titles_circles)


# Функція для візуалізації моделей логістичної регресії на всіх ознаках
def plot_iris_models_comparison(models, titles):
    # Зменшення розмірності до 2D за допомогою PCA
    pca = PCA(n_components=2)
    X_iris_pca = pca.fit_transform(X_train_iris)

    for model, title in zip(models, titles):
        plt.figure(figsize=(8, 6))

        # Визначити межі
        x_min, x_max = X_iris_pca[:, 0].min() - 1, X_iris_pca[:, 0].max() + 1
        y_min, y_max = X_iris_pca[:, 1].min() - 1, X_iris_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        # Створюємо новий масив, який складається з координат у PCA
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Прогнозування для кожної точки на сітці в зменшеному просторі
        Z_mesh = model.predict(pca.inverse_transform(grid_points))
        Z_mesh = Z_mesh.reshape(xx.shape)

        plt.contourf(xx, yy, Z_mesh, alpha=0.3, cmap='viridis')

        # Відобразити дані
        scatter = plt.scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_train_iris, edgecolors='k', cmap='viridis')
        plt.title(title)
        plt.xlabel('Перша головна складова')
        plt.ylabel('Друга головна складова')

        plt.tight_layout()
        plt.show()

# Візуалізуємо моделі для набору "ірис"
models_iris = [
    simple_log_reg_iris,
    multi_log_reg_iris,
    simple_log_no_reg_iris,
    multi_log_no_reg_iris
]

titles_iris = [
    "Проста логістична регресія (Регуляризація)",
    "Поліноміальна логістична регресія (Регуляризація)",
    "Проста логістична регресія (Без регуляризації)",
    "Поліноміальна логістична регресія (Без регуляризації)"
]

plot_iris_models_comparison(models_iris, titles_iris)


# 5 завдання - Виконати прогнози на основi побудованих моделей.
# Виконання прогнозів на валідаційних даних
y_val_pred_circles_simple_reg = simple_log_reg_circles.predict(X_val_circles)
y_val_pred_circles_multi_reg = multi_log_reg_circles.predict(X_val_circles)
y_val_pred_circles_simple_no_reg = simple_log_no_reg_circles.predict(X_val_circles)
y_val_pred_circles_multi_no_reg = multi_log_no_reg_circles.predict(X_val_circles)
y_val_pred_iris_simple_reg = simple_log_reg_iris.predict(X_val_iris)
y_val_pred_iris_multi_reg = multi_log_reg_iris.predict(X_val_iris)
y_val_pred_iris_simple_no_reg = simple_log_no_reg_iris.predict(X_val_iris)
y_val_pred_iris_multi_no_reg = multi_log_no_reg_iris.predict(X_val_iris)


y_train_pred_circles_simple_reg = simple_log_reg_circles.predict(X_train_circles)
y_train_pred_circles_multi_reg = multi_log_reg_circles.predict(X_train_circles)
y_train_pred_circles_simple_no_reg = simple_log_no_reg_circles.predict(X_train_circles)
y_train_pred_circles_multi_no_reg = multi_log_no_reg_circles.predict(X_train_circles)
y_train_pred_iris_simple_reg = simple_log_reg_iris.predict(X_train_iris)
y_train_pred_iris_multi_reg = multi_log_reg_iris.predict(X_train_iris)
y_train_pred_iris_simple_no_reg = simple_log_no_reg_iris.predict(X_train_iris)
y_train_pred_iris_multi_no_reg = multi_log_no_reg_iris.predict(X_train_iris)


# 6 завдання - Для кожної з моделей оцiнити, чи має мiсце перенавчання.
print("\nЗавдання 6")
train_accuracy_circle_simple_reg = accuracy_score(y_train_circles, y_train_pred_circles_simple_reg)
val_accuracy_circle_simple_reg = accuracy_score(y_val_circles, y_val_pred_circles_simple_reg)
train_accuracy_iris_simple_reg = accuracy_score(y_train_iris, y_train_pred_iris_simple_reg)
val_accuracy_iris_simple_reg = accuracy_score(y_val_iris, y_val_pred_iris_simple_reg)
train_accuracy_circle_multi_reg = accuracy_score(y_train_circles, y_train_pred_circles_multi_reg)
val_accuracy_circle_multi_reg = accuracy_score(y_val_circles, y_val_pred_circles_multi_reg)
train_accuracy_iris_multi_reg = accuracy_score(y_train_iris, y_train_pred_iris_multi_reg)
val_accuracy_iris_multi_reg = accuracy_score(y_val_iris, y_val_pred_iris_multi_reg)

train_accuracy_circle_simple_no_reg = accuracy_score(y_train_circles, y_train_pred_circles_simple_no_reg)
val_accuracy_circle_simple_no_reg = accuracy_score(y_val_circles, y_val_pred_circles_simple_no_reg)
train_accuracy_iris_simple_no_reg = accuracy_score(y_train_iris, y_train_pred_iris_simple_no_reg)
val_accuracy_iris_simple_no_reg = accuracy_score(y_val_iris, y_val_pred_iris_simple_no_reg)
train_accuracy_circle_multi_no_reg = accuracy_score(y_train_circles, y_train_pred_circles_multi_no_reg)
val_accuracy_circle_multi_no_reg = accuracy_score(y_val_circles, y_val_pred_circles_multi_no_reg)
train_accuracy_iris_multi_no_reg = accuracy_score(y_train_iris, y_train_pred_iris_multi_no_reg)
val_accuracy_iris_multi_no_reg = accuracy_score(y_val_iris, y_val_pred_iris_multi_no_reg)

print(f"\nТочність для навчального набору (Сircle, simple, reg): {train_accuracy_circle_simple_reg:.5f}")
print(f"Точність для валідаційного набору (Сircle, simple, reg): {val_accuracy_circle_simple_reg:.5f}")
print("\nЗвіт по метриках для набору (Сircle, simple, reg):")
print(classification_report(y_val_circles, y_val_pred_circles_simple_reg))

print(f"\nТочність для навчального набору (Сircle, multi, reg): {train_accuracy_circle_multi_reg:.5f}")
print(f"Точність для валідаційного набору (Сircle, multi, reg): {val_accuracy_circle_multi_reg:.5f}")
print("\nЗвіт по метриках для набору (Сircle, multi, reg):")
print(classification_report(y_val_circles, y_val_pred_circles_multi_reg))

print(f"\nТочність для навчального набору (Сircle, simple, no reg): {train_accuracy_circle_simple_no_reg:.5f}")
print(f"Точність для валідаційного набору (Сircle, simple, no reg): {val_accuracy_circle_simple_no_reg:.5f}")
print("\nЗвіт по метриках для набору (Сircle, simple, no reg):")
print(classification_report(y_val_circles, y_val_pred_circles_simple_no_reg))

print(f"\nТочність для навчального набору (Сircle, multi, no reg): {train_accuracy_circle_multi_no_reg:.5f}")
print(f"Точність для валідаційного набору (Сircle, multi, no reg): {val_accuracy_circle_multi_no_reg:.5f}")
print("\nЗвіт по метриках для набору (Сircle, multi, no reg):")
print(classification_report(y_val_circles, y_val_pred_circles_multi_no_reg))

print(f"\nТочність для навчального набору (Iris, simple, reg): {train_accuracy_iris_simple_reg:.5f}")
print(f"Точність для валідаційного набору (Iris, simple, reg): {val_accuracy_iris_simple_reg:.5f}")
print("\nЗвіт по метриках для набору (Iris, simple, reg):")
print(classification_report(y_val_iris, y_val_pred_iris_simple_reg))

print(f"\nТочність для навчального набору (Iris, multi, reg): {train_accuracy_iris_multi_reg:.5f}")
print(f"Точність для валідаційного набору (Iris, multi, reg): {val_accuracy_iris_multi_reg:.5f}")
print("\nЗвіт по метриках для набору (Iris, multi, reg):")
print(classification_report(y_val_iris, y_val_pred_iris_multi_reg))

print(f"\nТочність для навчального набору (Iris, simple, no reg): {train_accuracy_iris_simple_no_reg:.5f}")
print(f"Точність для валідаційного набору (Iris, simple, no reg): {val_accuracy_iris_simple_no_reg:.5f}")
print("\nЗвіт по метриках для набору (Iris, simple, no reg):")
print(classification_report(y_val_iris, y_val_pred_iris_simple_no_reg))

print(f"\nТочність для навчального набору (Iris, multi, no reg): {train_accuracy_iris_multi_no_reg:.5f}")
print(f"Точність для валідаційного набору (Iris, multi, no reg): {val_accuracy_iris_multi_no_reg:.5f}")
print("\nЗвіт по метриках для набору (Iris, multi, no reg):")
print(classification_report(y_val_iris, y_val_pred_iris_multi_no_reg))


# 7 завдання - Розрахувати додатковi результати моделей, наприклад, апостерiорнi iмовiрностi або iншi (згiдно з варiантом).
print("\nЗавдання 7")
test_example_circles_simple_reg = X_val_circles[0].reshape(1, -1)
proba_test_circles_simple_reg = simple_log_reg_circles.predict_proba(test_example_circles_simple_reg)
test_example_circles_simple_no_reg = X_val_circles[0].reshape(1, -1)
proba_test_circles_simple_no_reg = simple_log_no_reg_circles.predict_proba(test_example_circles_simple_no_reg)
test_example_circles_multi_reg = X_val_circles[0].reshape(1, -1)
proba_test_circles_multi_reg = multi_log_reg_circles.predict_proba(test_example_circles_multi_reg)
test_example_circles_multi_no_reg = X_val_circles[0].reshape(1, -1)
proba_test_circles_multi_no_reg = multi_log_no_reg_circles.predict_proba(test_example_circles_multi_no_reg)

test_example_iris_simple_reg = X_val_iris[0].reshape(1, -1)
proba_test_iris_simple_reg = simple_log_reg_iris.predict_proba(test_example_iris_simple_reg)
test_example_iris_simple_no_reg = X_val_iris[0].reshape(1, -1)
proba_test_iris_simple_no_reg = simple_log_no_reg_iris.predict_proba(test_example_iris_simple_no_reg)
test_example_iris_multi_reg = X_val_iris[0].reshape(1, -1)
proba_test_iris_multi_reg = multi_log_reg_iris.predict_proba(test_example_iris_multi_reg)
test_example_iris_multi_no_reg = X_val_iris[0].reshape(1, -1)
proba_test_iris_multi_no_reg = multi_log_no_reg_iris.predict_proba(test_example_iris_multi_no_reg)

print(f"Тестовий приклад для (Circle, simple, reg): {test_example_circles_simple_reg}")
print(f"Апостеріорні ймовірності для тестового прикладу (Circle, simple, reg): {proba_test_circles_simple_reg}")
print(f"Тестовий приклад для (Circle, simple, no reg): {test_example_circles_simple_no_reg}")
print(f"Апостеріорні ймовірності для тестового прикладу (Circle, simple, no reg): {proba_test_circles_simple_no_reg}")
print(f"Тестовий приклад для (Circle, multi, reg): {test_example_circles_multi_reg}")
print(f"Апостеріорні ймовірності для тестового прикладу (Circle, multi, reg): {proba_test_circles_multi_reg}")
print(f"Тестовий приклад для (Circle, multi, no reg): {test_example_circles_multi_no_reg}")
print(f"Апостеріорні ймовірності для тестового прикладу (Circle, multi, no reg): {proba_test_circles_multi_no_reg}")

print(f"Тестовий приклад для (Iris, simple, reg): {test_example_iris_simple_reg}")
print(f"Апостеріорні ймовірності для тестового прикладу (Iris, simple, reg): {proba_test_iris_simple_reg}")
print(f"Тестовий приклад для (Iris, simple, no reg): {test_example_iris_simple_no_reg}")
print(f"Апостеріорні ймовірності для тестового прикладу (Iris, simple, no reg): {proba_test_iris_simple_no_reg}")
print(f"Тестовий приклад для (Iris, multi, reg): {test_example_iris_multi_reg}")
print(f"Апостеріорні ймовірності для тестового прикладу (Iris, multi, reg): {proba_test_iris_multi_reg}")
print(f"Тестовий приклад для (Iris, multi, no reg): {test_example_iris_multi_no_reg}")
print(f"Апостеріорні ймовірності для тестового прикладу (Iris, multi, no reg): {proba_test_iris_multi_no_reg}")


# 10 завдання - В задачах регресiї розрахувати для кожної моделi наступнi критерiї якостi, окремо на навчальнiй та валiдацiйнiй множинах:
# • коефiцiєнт детермiнацiї R2
# • помилки RMSE, MAE та MAPE.
print("\nЗавдання 10")
def evaluate_regression_model(y_true, y_pred, model_name=""):
    # R²
    r2 = r2_score(y_true, y_pred)

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error) з ігноруванням нульових значень в y_true
    epsilon = 1e-10  # дуже маленьке значення, яке додаємо до y_true для уникнення ділення на нуль
    non_zero_indices = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

    # Виведення результатів
    print(f"\nОцінка моделі {model_name}:")
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"MAPE: {mape:.3f}%")


def evaluate_models(models):
    for model_name, (model, X_train, y_train, X_val, y_val) in models.items():
        # Прогноз на навчальному наборі
        y_train_pred = model.predict(X_train)
        print(f"\nОцінка на навчальній множині для {model_name}:")
        evaluate_regression_model(y_train, y_train_pred, model_name)

        # Прогноз на валідаційному наборі
        y_val_pred = model.predict(X_val)
        print(f"\nОцінка на валідаційній множині для {model_name}:")
        evaluate_regression_model(y_val, y_val_pred, model_name)


# Словник моделей
models = {
    "Проста логістична регресія (регуляризація, кільця)": (
        simple_log_reg_circles, X_train_circles, y_train_circles, X_val_circles, y_val_circles),
    "Поліноміальна логістична регресія (регуляризація, кільця)": (
        multi_log_reg_circles, X_train_circles, y_train_circles, X_val_circles, y_val_circles),
    "Проста логістична регресія (без регуляризації, кільця)": (
        simple_log_no_reg_circles, X_train_circles, y_train_circles, X_val_circles, y_val_circles),
    "Поліноміальна логістична регресія (без регуляризації, кільця)": (
        multi_log_no_reg_circles, X_train_circles, y_train_circles, X_val_circles, y_val_circles),
    "Проста логістична регресія (регуляризація, Iris)": (
        simple_log_reg_iris, X_train_iris, y_train_iris, X_val_iris, y_val_iris),
    "Поліноміальна логістична регресія (регуляризація, Iris)": (
        multi_log_reg_iris, X_train_iris, y_train_iris, X_val_iris, y_val_iris),
    "Проста логістична регресія (без регуляризації, Iris)": (
        simple_log_no_reg_iris, X_train_iris, y_train_iris, X_val_iris, y_val_iris),
    "Поліноміальна логістична регресія (без регуляризації, Iris)": (
        multi_log_no_reg_iris, X_train_iris, y_train_iris, X_val_iris, y_val_iris)
}

# Виклик функції для оцінки моделей
evaluate_models(models)


# 11 завдання - Спробувати виконати решiтчастий пошук (grid search) для пiдбору гiперпараметрiв моделей.
print("\nЗавдання 11")
warnings.filterwarnings("ignore")
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Регуляризація
    'solver': ['liblinear', 'lbfgs', 'newton-cg', 'saga'],  # Методи оптимізації
    'penalty': ['l1', 'l2', 'elasticnet'],  # Тип регуляризації
    'multi_class': ['ovr', 'multinomial'],  # Тип класифікації
    'max_iter': [100, 200, 500]  # Максимальна кількість ітерацій
}

# Решітчастий пошук для простої логістичної регресії
grid_search_circle = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search_circle.fit(X_train_circles, y_train_circles)
print("Найкращі параметри для простої логістичної регресії (Circle): ", grid_search_circle.best_params_)
print("Найкраща точність: ", grid_search_circle.best_score_)

# Решітчастий пошук для простої логістичної регресії
grid_search_iris = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search_iris.fit(X_train_iris, y_train_iris)
print("Найкращі параметри для простої логістичної регресії (Iris): ", grid_search_iris.best_params_)
print("Найкраща точність: ", grid_search_iris.best_score_)


# 12. Зробити висновки про якiсть роботи моделей на дослiджених даних. На основi критерiїв якостi
# спробувати обрати найкращу модель.



# 13 завдання - Навчити моделi на пiдмножинах навчальних даних. Оцiнити, наскiльки
# розмiр навчальної множини впливає на якiсть моделi.
print("\nЗавдання 13")
def evaluate_on_different_sample_sizes(X, y, sample_sizes, model_class, **kwargs):
    results = []
    for size in sample_sizes:
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=size, random_state=42)



        # Initialize the model
        model = model_class(**kwargs)

        # Train the model
        model.fit(X_train, y_train)

        # Predict on validation set
        y_val_pred = model.predict(X_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_val_pred)
        results.append((size, accuracy))

        # Print classification report
        print(f"Sample Size: {size:.0%} - Accuracy: {accuracy:.4f}")
        print(classification_report(y_val, y_val_pred))

    return results

# Define sample sizes to test
sample_sizes = [0.1, 0.2, 0.5, 0.99]  # 10%, 20%, 50%, 99%

# Evaluate models on the circles dataset
print("Evaluating Circles Dataset (simple):")
evaluate_on_different_sample_sizes(X, y, sample_sizes, LogisticRegression)

print("Evaluating Circles Dataset (multi):")
evaluate_on_different_sample_sizes(X, y, sample_sizes, LogisticRegression, multi_class='multinomial',
                                   solver='lbfgs')

# Evaluate models on the Iris dataset
print("\nEvaluating Iris Dataset:")
evaluate_on_different_sample_sizes(X_iris, y_iris, sample_sizes, LogisticRegression)

print("\nEvaluating Iris Dataset (multi):")
evaluate_on_different_sample_sizes(X_iris, y_iris, sample_sizes, LogisticRegression, multi_class='multinomial',
                                   solver='lbfgs')


# 14. Кожний варiант мiстить два набори даних. Дослiдити обидва набори за
# наведеними вище етапами. Можна обрати власний набiр даних (повiдо-
# мивши попередньо про це викладача), наприклад, з цiкавої вам практи-
# чної задачi. Для кожного набору спробувати пiдiбрати найкращу модель.