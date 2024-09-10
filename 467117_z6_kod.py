import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Wczytanie danych
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')

# Model z najlepszymi hiperparametrami
best_rf_model = RandomForestRegressor(
    bootstrap=False,
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=350,
    random_state=42
)
best_rf_model.fit(X_train, y_train.values.ravel())

# Predykcja na zbiorze testowym i zapisanie do pliku csv
y_pred = best_rf_model.predict(X_test)
predictions = pd.DataFrame({'Id': X_test.index, 'Expected': y_pred})
predictions.to_csv('predykcja.csv', index=False)


