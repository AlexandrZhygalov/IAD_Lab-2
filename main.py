import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import make_circles, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_curve, auc, precision_recall_curve,
                             mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error)


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
# Функція для побудови границь рішень
"""
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
    plt.title(title)
    plt.show()

# Візуалізація для набору даних "кільця"
plot_decision_boundary(simple_log_reg_circles, X_train_circles, y_train_circles, "Проста логістична регресія з регуляризацією (кільця)")
plot_decision_boundary(simple_log_no_reg_circles, X_train_circles, y_train_circles, "Проста логістична регресія без регуляризації (кільця)")

plot_decision_boundary(multi_log_reg_circles, X_train_circles, y_train_circles, "Поліноміальна логістична регресія з регуляризацією (кільця)")
plot_decision_boundary(multi_log_no_reg_circles, X_train_circles, y_train_circles, "Поліноміальна логістична регресія без регуляризації (кільця)")


def plot_decision_boundary_2(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Отримуємо ймовірності для кожного класу
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis', marker='o')
    plt.title(title)
    plt.xlabel('Довжина чашолистка')
    plt.ylabel('Ширина чашолистка')
    plt.colorbar(ticks=[0, 1, 2], label='Класи')
    plt.show()

# # Для набору Iris залишимо тільки дві ознаки, щоб можна було графічно представити границі рішень
X_iris_2d = X_train_iris[:, :2]  # Вибираємо перші дві ознаки для зручності візуалізації

plot_decision_boundary_2(simple_log_reg_iris, X_iris_2d, y_train_iris, "Проста логістична регресія з регуляризацією (Iris)")
plot_decision_boundary_2(simple_log_no_reg_iris, X_iris_2d, y_train_iris, "Проста логістична регресія без регуляризації (Iris)")

plot_decision_boundary_2(multi_log_reg_iris, X_iris_2d, y_train_iris, "Поліноміальна логістична регресія з регуляризацією (Iris)")
plot_decision_boundary_2(multi_log_no_reg_iris, X_iris_2d, y_train_iris, "Поліноміальна логістична регресія без регуляризації (Iris)")

"""
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
"""
simple_log_no_reg_circles.fit(X_train_circles, y_train_circles)
simple_log_no_reg_iris.fit(X_train_iris, y_train_iris)
multi_log_no_reg_circles.fit(X_train_circles, y_train_circles)
multi_log_no_reg_iris.fit(X_train_iris, y_train_iris)
simple_log_reg_circles.fit(X_train_circles, y_train_circles)
simple_log_reg_iris.fit(X_train_iris, y_train_iris)
multi_log_reg_circles.fit(X_train_circles, y_train_circles)
multi_log_reg_iris.fit(X_train_iris, y_train_iris)


"""


# Виконання прогнозів на валідаційних даних
y_pred_circles_simple_reg = simple_log_reg_circles.predict(X_val_circles)
y_pred_circles_multi_reg = multi_log_reg_circles.predict(X_val_circles)
y_pred_circles_simple_no_reg = simple_log_no_reg_circles.predict(X_val_circles)
y_pred_circles_multi_no_reg = multi_log_no_reg_circles.predict(X_val_circles)
y_pred_iris_simple_reg = simple_log_reg_iris.predict(X_val_iris)
y_pred_iris_multi_reg = multi_log_reg_iris.predict(X_val_iris)
y_pred_iris_simple_no_reg = simple_log_no_reg_iris.predict(X_val_iris)
y_pred_iris_multi_no_reg = multi_log_no_reg_iris.predict(X_val_iris)


# 6 завдання - Для кожної з моделей оцiнити, чи має мiсце перенавчання.





# 7 завдання - Для кожної з моделей оцiнити, чи має мiсце перенавчання.
# Апостеріорні ймовірності для тестового прикладу (make_circles)
print("Апостеріорні ймовірності (простий логістичний регресор):", proba_simple_circles[0])
print("Апостеріорні ймовірності (поліноміальний логістичний регресор):", proba_poly_circles[0])
# Апостеріорні ймовірності для тестового прикладу (Iris)
print("Апостеріорні ймовірності (простий логістичний регресор):", proba_simple_iris[0])
print("Апостеріорні ймовірності (поліноміальний логістичний регресор):", proba_poly_iris[0])








"""
# Оцінка якості моделі
r2_circles = r2_score(y_val_circles, y_pred_circles)
rmse_circles = np.sqrt(mean_squared_error(y_val_circles, y_pred_circles))
mae_circles = mean_absolute_error(y_val_circles, y_pred_circles)
mape_circles = mean_absolute_percentage_error(y_val_circles, y_pred_circles)

# Виводимо результати
print("Результати для make_circles:")
print(f"R²: {r2_circles:.4f}, RMSE: {rmse_circles:.4f}, MAE: {mae_circles:.4f}, MAPE: {mape_circles:.4f}%")

# Оцінка якості моделі
r2_iris = r2_score(y_val_iris, y_pred_iris)
rmse_iris = np.sqrt(mean_squared_error(y_val_iris, y_pred_iris))
mae_iris = mean_absolute_error(y_val_iris, y_pred_iris)
mape_iris = mean_absolute_percentage_error(y_val_iris, y_pred_iris)

# Виводимо результати
print("Результати для load_iris:")
print(f"R²: {r2_iris:.4f}, RMSE: {rmse_iris:.4f}, MAE: {mae_iris:.4f}, MAPE: {mape_iris:.4f}%")
"""

