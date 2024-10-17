import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_circles, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_curve, auc, precision_recall_curve,
                             mean_squared_error, mean_absolute_error, r2_score)


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
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
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
