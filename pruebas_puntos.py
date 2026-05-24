import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Columnas
cols_fecha = ['year', 'month', 'day']
cols_elo = ['elo_h', 'elo_a']
cols_equipos = []


# Preparar datos con One-Hot Encoding
def preparar_datos_ohe(df, cols_equipos):
    df_copy = df.copy()
    df_copy['game_date'] = pd.to_datetime(df_copy['game_date'])
    df_copy['year'] = df_copy['game_date'].dt.year
    df_copy['month'] = df_copy['game_date'].dt.month
    df_copy['day'] = df_copy['game_date'].dt.day
    df_copy = pd.get_dummies(df_copy, columns=['team_abbreviation_home', 'team_abbreviation_away'], dtype=float)

    if cols_equipos == []:
        cols_equipos = [c for c in df_copy.columns if 'team_abbreviation_home_' in c or 'team_abbreviation_away_' in c]
    return df_copy, cols_equipos


# Escalar datos
def escalar_datos(df, df_test, cols_no_escalables, cols_escalables):
    scaler = StandardScaler()
    scaler.fit(df[cols_escalables])

    df_escalado = scaler.transform(df[cols_escalables])
    df_test_escalado = scaler.transform(df_test[cols_escalables])

    data_entrada = np.hstack([np.array(df[cols_no_escalables]), df_escalado])
    data_entrada_test = np.hstack([np.array(df_test[cols_no_escalables]), df_test_escalado])
    return data_entrada, data_entrada_test


# Preparar salida para regresión de puntuaciones
def preparar_datos_salida_puntuacion(df):
    return df[['pts_home', 'pts_away']].values.astype(float)


# Crear modelo de regresión
def crear_modelo_puntuacion(n_input):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=[n_input]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='linear')  # salida continua
    ])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mean_squared_error'
    )
    return modelo


# Evaluar modelo de puntuación
def evaluar_modelo_puntuacion(modelo, entrada, salida_real, nombre_modelo):
    predicciones = modelo.predict(entrada)
    mse = mean_squared_error(salida_real, predicciones)
    mae = mean_absolute_error(salida_real, predicciones)

    print(f"\n--- Resultados {nombre_modelo} ---")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    return mse, mae


# Guardar resultados
def guardar_resultados_csv_puntuacion(df, modelo, entrada, nombre_archivo):
    if not os.path.exists('resultados2'):
        os.makedirs('resultados2')

    df = df.copy()
    df = df[df['season_id'].astype(str).str[-4:].astype(int) > 2017]
    df = df[['season_id', 'game_date', 'team_name_home', 'team_name_away', 'pts_home', 'pts_away']]
    predicciones = modelo.predict(entrada)
    df['pred_pts_home'] = predicciones[:, 0]
    df['pred_pts_away'] = predicciones[:, 1]
    df.to_csv('resultados2/' + nombre_archivo, index=False)


# Lectura y preparación de datos
df_partidos = pd.read_csv('csv_red/partidos.csv')
df_partidos, cols_equipos = preparar_datos_ohe(df_partidos, cols_equipos)

partidos = df_partidos[df_partidos['season_id'].astype(str).str[-4:].astype(int) <= 2017]
partidos_test = df_partidos[df_partidos['season_id'].astype(str).str[-4:].astype(int) > 2017]

data_entrada, data_entrada_test = escalar_datos(partidos, partidos_test, cols_equipos, cols_fecha)
data_salida, data_salida_test = preparar_datos_salida_puntuacion(partidos), preparar_datos_salida_puntuacion(
    partidos_test)

# Crear y entrenar modelo
modelo = crear_modelo_puntuacion(data_entrada.shape[1])
history = modelo.fit(
    data_entrada, data_salida,
    epochs=1000,
    #verbose=1,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )]
)

# Evaluación
mse, mae = evaluar_modelo_puntuacion(modelo, data_entrada_test, data_salida_test, "Modelo Puntuación")

# Guardar resultados
guardar_resultados_csv_puntuacion(partidos_test, modelo, data_entrada_test, 'resultados_modelo_puntuacion.csv')