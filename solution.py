import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

print("🚀 Старт продвинутого ML-решения для задачи 'Тишина АЗ-26'...")

# 1. Проверяем наличие файлов телеметрии
train_path = "telemetry_train.csv"
test_path = "telemetry_test.csv"

if not os.path.exists(test_path):
    raise FileNotFoundError(f"❌ Ошибка: Файл {test_path} не найден! Загрузите его в рабочую директорию.")

# 2. Загрузка тестовых данных
test = pd.read_csv(test_path)
print(f"📊 Тестовые данные успешно загружены: {test.shape[0]} строк, {test.shape[1]} колонок.")

# Определяем список датчиков, исключая служебные поля
SENS = [c for c in test.columns if c not in ["timestamp_id", "session_id", "mode", "dt"]]

# Заполнение пропусков (forward fill + замена оставшихся NaN на 0)
test[SENS] = test[SENS].ffill().fillna(0)

# 3. Feature Engineering (Создание признаков для модели ИИ)
print("🛠 Генерируем признаки для модели...")
def generate_features(df):
    features_list = []
    
    # Отклонение текущих значений от медианы по сессии (логика базового решения)
    usual = df.groupby("session_id")[SENS].transform("median")
    deviation = (df[SENS] - usual).abs()
    deviation.columns = [f"{c}_dev" for c in SENS]
    features_list.append(deviation)
    
    # Скользящее стандартное отклонение по окну из 5 замеров для оценки стабильности датчика
    rolling_std = df.groupby("session_id")[SENS].rolling(window=5, min_periods=1).std().reset_index(level=0, drop=True)
    rolling_std.columns = [f"{c}_roll_std" for c in SENS]
    features_list.append(rolling_std)
    
    # Исходные отфильтрованные датчики
    features_list.append(df[SENS])
    
    # Объединяем матрицы признаков
    X = pd.concat(features_list, axis=1)
    X = X.fillna(0)
    return X

X_test = generate_features(test)
print(f"✅ Матрица признаков готова. Итоговая размерность: {X_test.shape}")

# 4. Шкалирование признаков (RobustScaler устойчив к сильным выбросам и аномалиям)
scaler = RobustScaler()

# 5. Обучение модели машинного обучения
if os.path.exists(train_path):
    print("🧠 Найден файл telemetry_train.csv! Запускаем обучение в режиме Semi-supervised...")
    train = pd.read_csv(train_path)
    train[SENS] = train[SENS].ffill().fillna(0)
    X_train = generate_features(train)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Строим изоляционный лес на основе обучающей выборки
    model = IsolationForest(n_estimators=150, contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled)
else:
    print("⚠️ Файл train не найден. Обучаем модель Isolation Forest напрямую на Test в режиме Unsupervised...")
    X_test_scaled = scaler.fit_transform(X_test)
    
    model = IsolationForest(n_estimators=150, contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X_test_scaled)

# 6. Расчет финальных скоров аномальности
print("🔮 Рассчитываем вероятности аномалий...")
# Инвертируем decision_function: чем ниже значение функции, тем более аномален объект
anomaly_scores = -model.decision_function(X_test_scaled)

# Нормализуем значения в диапазон [0, 1] для получения anomaly_prob
min_s = anomaly_scores.min()
max_s = anomaly_scores.max()
anomaly_prob = (anomaly_scores - min_s) / (max_s - min_s + 1e-9)

# 7. Формирование финального файла ответов
sub = pd.DataFrame({"timestamp_id": test.timestamp_id, "anomaly_prob": anomaly_prob})
sub.to_csv("submission.csv", index=False)

print("🎉 Успех! Файл submission.csv успешно сгенерирован и готов к отправке.")
