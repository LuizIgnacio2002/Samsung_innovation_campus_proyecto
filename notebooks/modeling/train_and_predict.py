import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Leer datos
print("Cargando datos...")
df = pd.read_csv('../data/03_casos/df_52_cleaned.csv')
print(f"Datos cargados: {df.shape}")

# Variables principales (sin incluir el target)
variables_principales = ['ano', 'semana', 'ira_no_neumonia', 'neumonias_men5', 'neumonias_60mas',
                         'hospitalizados_60mas', 'defunciones_men5', 'defunciones_60mas']

# Target
target = 'hospitalizados_men5'

# Filtrar datos de 2006 a 2023 para entrenamiento
df_train = df[(df['ano'] >= 2006) & (df['ano'] <= 2023)].copy()

print(f"\nDatos de entrenamiento: {len(df_train)} registros")
print(f"Periodo: {df_train['ano'].min()}-{df_train['ano'].max()}")
print(f"Shape: {df_train.shape}")

# Separar features y target
X = df_train[variables_principales]
y = df_train[target]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nEntrenando modelo CatBoost...")
# Entrenar modelo CatBoost
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=100
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

print("\n✓ Modelo entrenado exitosamente!")

# Evaluar el modelo
y_pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

print(f"\nMétricas del modelo:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# Generar fechas futuras (2024-2025)
print("\nGenerando fechas futuras...")
# Calcular promedios históricos por semana para las variables predictoras
promedios_por_semana = df_train.groupby('semana')[
    ['ira_no_neumonia', 'neumonias_men5', 'neumonias_60mas',
     'hospitalizados_60mas', 'defunciones_men5', 'defunciones_60mas']
].mean()

# Crear DataFrame para predicciones futuras
years_future = [2024, 2025]
semanas = list(range(1, 53))  # 52 semanas por año

future_data = []
for year in years_future:
    for semana in semanas:
        row = {'ano': year, 'semana': semana}
        # Usar promedios históricos para cada semana del año
        for col in ['ira_no_neumonia', 'neumonias_men5', 'neumonias_60mas',
                    'hospitalizados_60mas', 'defunciones_men5', 'defunciones_60mas']:
            row[col] = promedios_por_semana.loc[semana, col]
        future_data.append(row)

df_future = pd.DataFrame(future_data)
print(f"Datos futuros generados: {len(df_future)} registros (años 2024-2025)")

# Realizar predicciones para 2024-2025
print("\nRealizando predicciones...")
X_future = df_future[variables_principales]
predicciones = model.predict(X_future)

# Agregar predicciones al DataFrame futuro
df_future['hospitalizados_men5_prediccion'] = predicciones

print(f"✓ Predicciones realizadas: {len(predicciones)} valores")
print("\nPrimeras 10 predicciones:")
print(df_future[['ano', 'semana', 'hospitalizados_men5_prediccion']].head(10))
print("\nEstadísticas de predicciones:")
print(df_future['hospitalizados_men5_prediccion'].describe())

# Guardar predicciones en CSV
output_path = '../data/03_casos/df_52_cleaned_predicciones.csv'
df_future.to_csv(output_path, index=False)

print(f"\n✓ Predicciones guardadas exitosamente en: {output_path}")
print(f"  - Total de registros: {len(df_future)}")
print(f"  - Años incluidos: {df_future['ano'].min()} - {df_future['ano'].max()}")
print(f"  - Columnas: {list(df_future.columns)}")
