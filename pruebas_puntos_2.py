import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# =========================
# COLUMNAS
# =========================
cols_fecha = ['year', 'month', 'day']

cols_elo = [
    'elo_h','elo_a',
    'elo_h_ofg','elo_a_ofg',
    'elo_h_dfg','elo_a_dfg',
    'elo_h_ofg3','elo_a_ofg3',
    'elo_h_dfg3','elo_a_dfg3',
    'elo_diff','elo_ofg_diff','elo_dfg_diff',
    'elo_ofg3_diff','elo_dfg3_diff',
    'playoffs'
]

# =========================
# PREPARAR DATOS
# =========================
def preparar_datos_ohe(df, cols_equipos):
    df_copy = df.copy()
    df_copy['game_date'] = pd.to_datetime(df_copy['game_date'])
    df_copy['year'] = df_copy['game_date'].dt.year
    df_copy['month'] = df_copy['game_date'].dt.month
    df_copy['day'] = df_copy['game_date'].dt.day

    df_copy = pd.get_dummies(
        df_copy,
        columns=['team_abbreviation_home','team_abbreviation_away'],
        dtype=float
    )

    if cols_equipos == []:
        cols_equipos = [
            c for c in df_copy.columns
            if 'team_abbreviation_home_' in c or 'team_abbreviation_away_' in c
        ]

    return df_copy, cols_equipos

# =========================
# ESCALAR DATOS
# =========================
def escalar_datos(df, df_test, cols_no_escalables, cols_escalables):
    scaler = StandardScaler()
    scaler.fit(df[cols_escalables])

    df_escalado = scaler.transform(df[cols_escalables])
    df_test_escalado = scaler.transform(df_test[cols_escalables])

    data_entrada = np.hstack([np.array(df[cols_no_escalables]), df_escalado])
    data_entrada_test = np.hstack([np.array(df_test[cols_no_escalables]), df_test_escalado])

    return data_entrada, data_entrada_test

# =========================
# SALIDA
# =========================
def preparar_datos_salida_puntuacion(df):
    return df[['pts_home','pts_away']].values.astype(float)

# =========================
# MODELO
# =========================
def crear_modelo_puntuacion(n_input):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=[n_input]),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='linear')
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mean_absolute_error'
    )

    return modelo

# =========================
# EVALUACIÓN
# =========================
def evaluar_modelo_puntuacion(modelo, entrada, salida_real, nombre_modelo):
    predicciones = modelo.predict(entrada)

    mse = mean_squared_error(salida_real, predicciones)
    mae = mean_absolute_error(salida_real, predicciones)

    print(f"\n--- Resultados {nombre_modelo} ---")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")

    return mse, mae

# =========================
# GUARDAR RESULTADOS
# =========================
def guardar_resultados_csv_puntuacion(df, modelo, entrada, nombre_archivo):
    if not os.path.exists('resultados2'):
        os.makedirs('resultados2')

    df = df.copy()
    df = df[df['season_id'].astype(str).str[-4:].astype(int) > 2017]

    df = df[['season_id','game_date','team_name_home','team_name_away','pts_home','pts_away']]

    predicciones = modelo.predict(entrada)
    df['pred_pts_home'] = predicciones[:,0]
    df['pred_pts_away'] = predicciones[:,1]

    df.to_csv('resultados2/' + nombre_archivo, index=False)

# =========================
# CARGA DE DATOS (ELO3)
# =========================
df_partidos = pd.read_csv('csv_red/partidos_elo3.csv')

# =========================
# FEATURES ELO
# =========================
df_partidos['elo_diff'] = df_partidos['elo_h'] - df_partidos['elo_a']
df_partidos['elo_ofg_diff'] = df_partidos['elo_h_ofg'] - df_partidos['elo_a_ofg']
df_partidos['elo_dfg_diff'] = df_partidos['elo_h_dfg'] - df_partidos['elo_a_dfg']
df_partidos['elo_ofg3_diff'] = df_partidos['elo_h_ofg3'] - df_partidos['elo_a_ofg3']
df_partidos['elo_dfg3_diff'] = df_partidos['elo_h_dfg3'] - df_partidos['elo_a_dfg3']

# =========================
# ONE HOT
# =========================
df_partidos, cols_equipos = preparar_datos_ohe(df_partidos, [])

# =========================
# SPLIT TRAIN / TEST
# =========================
partidos = df_partidos[df_partidos['season_id'].astype(str).str[-4:].astype(int) <= 2017]
partidos_test = df_partidos[df_partidos['season_id'].astype(str).str[-4:].astype(int) > 2017]

# =========================
# DATOS DE ENTRADA
# =========================
data_entrada, data_entrada_test = escalar_datos(
    partidos,
    partidos_test,
    cols_equipos,
    cols_fecha + cols_elo
)

# =========================
# DATOS DE SALIDA
# =========================
data_salida = preparar_datos_salida_puntuacion(partidos)
data_salida_test = preparar_datos_salida_puntuacion(partidos_test)

# =========================
# ENTRENAMIENTO
# =========================
modelo = crear_modelo_puntuacion(data_entrada.shape[1])

history = modelo.fit(
    data_entrada,
    data_salida,
    epochs=1000,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )],
    verbose=1
)

# =========================
# EVALUACIÓN
# =========================
mse, mae = evaluar_modelo_puntuacion(
    modelo,
    data_entrada_test,
    data_salida_test,
    "Modelo con ELO3"
)


# =========================
# FUNCIÓN PARA EVALUAR GANADOR#############################################
# =========================
def evaluar_ganador(modelo, entrada, salida_real, nombre_modelo):
    """
    Compara la predicción de puntos del modelo y determina qué equipo gana.
    Devuelve el porcentaje de aciertos (accuracy) en predicción del ganador.
    """
    predicciones = modelo.predict(entrada)

    # Determinar ganador real y predicho
    ganador_real = (salida_real[:, 0] > salida_real[:, 1]).astype(int)  # 1 si gana home, 0 si gana away
    ganador_pred = (predicciones[:, 0] > predicciones[:, 1]).astype(int)

    # Calcular accuracy
    accuracy = np.mean(ganador_real == ganador_pred)

    print(f"\n--- Accuracy predicción ganador {nombre_modelo} ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy


# =========================
# EVALUACIÓN DE GANADOR####################################################################
# =========================
def evaluar_ganador(modelo, entrada, salida_real, nombre_modelo):
    predicciones = modelo.predict(entrada)
    # Decidir ganador: 1 si gana el local, 0 si gana el visitante
    pred_ganador = (predicciones[:, 0] > predicciones[:, 1]).astype(int)
    # Ganador real
    real_ganador = (salida_real[:, 0] > salida_real[:, 1]).astype(int)

    accuracy = (pred_ganador == real_ganador).mean()
    print(f"\n--- Predicción de ganador {nombre_modelo} ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# =========================
# PREDICCIÓN DE GANADOR###################################
# =========================
accuracy = evaluar_ganador(
    modelo,
    data_entrada_test,
    data_salida_test,
    "Modelo con ELO3"
)

# =========================
# GUARDAR RESULTADOS
# =========================
guardar_resultados_csv_puntuacion(
    partidos_test,
    modelo,
    data_entrada_test,
    'resultados_modelo_elo3_puntuacion.csv'
)