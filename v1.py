import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

cols_fecha = ['year', 'month', 'day']
cols_elo = ['elo_h', 'elo_a', 'elo_h_ofg', 'elo_a_ofg', 'elo_h_dfg', 'elo_a_dfg', 
            'elo_h_ofg3', 'elo_a_ofg3', 'elo_h_dfg3', 'elo_a_dfg3']
cols_equipos = []

#One-Hot Encoding de los equipos 
def preparar_datos_ohe(df, cols_equipos):
    df_copy = df.copy()
    # Extraer año, mes y día de la fecha del partido
    df_copy['game_date'] = pd.to_datetime(df_copy['game_date'])
    df_copy['year'] = df_copy['game_date'].dt.year
    df_copy['month'] = df_copy['game_date'].dt.month
    df_copy['day'] = df_copy['game_date'].dt.day

    # Creación de columnas One-Hot Encoding para los equipos
    df_copy = pd.get_dummies(df_copy, columns=['team_abbreviation_home','team_abbreviation_away'], dtype=float)
    
    # Nuevas columnas team_abbreviation_home_XXX y team_abbreviation_away_XXX
    if cols_equipos == []:
        cols_equipos = [c for c in df_copy.columns if 'team_abbreviation_home_' in c or 'team_abbreviation_away_' in c]
        
    return df_copy, cols_equipos

# Esclar datos para el modelo
def escalar_datos(df, df_test, cols_equipos, cols_escalables):
    scaler = StandardScaler()
   
    # Ajustar escalador solo con las columnas fecha y ELO de los datos de entrenamiento
    scaler.fit(df[cols_escalables])

    # Transformar sets
    df_escalado = scaler.transform(df[cols_escalables])
    df_test_escalado = scaler.transform(df_test[cols_escalables])
    # Combinar con las columnas de equipos 
    data_entrada = np.hstack([np.array(df[cols_equipos]), df_escalado])
    data_entrada_test = np.hstack([np.array(df_test[cols_equipos]), df_test_escalado])
    
    return data_entrada, data_entrada_test

# Partidos con datos básicos
df_partidos = pd.read_csv('csv_red/partidos.csv')
ultima_fecha = pd.to_datetime(df_partidos['game_date'].max())
df_partidos, cols_equipos = preparar_datos_ohe(df_partidos, cols_equipos)

# Filtrar partidos anteriores al último mes de datos para entrenamiento
partidos = df_partidos[df_partidos['game_date'] < ultima_fecha]# - pd.Timedelta(days=30)]
partidos_test = df_partidos[df_partidos['game_date'] >= ultima_fecha]# - pd.Timedelta(days=30)]
data_entrada, data_entrada_test = escalar_datos(partidos, partidos_test, cols_equipos, cols_fecha)
data_salida, data_salida_test = np.array(partidos[['pts_home', 'pts_away']]), np.array(partidos_test[['pts_home', 'pts_away']])

#Partidos con ELO1
df_partidos_elo1 = pd.read_csv('csv_red/partidos_elo1.csv')
df_partidos_elo1, cols_equipos = preparar_datos_ohe(df_partidos_elo1, cols_equipos)

partidos_elo1 = df_partidos_elo1[df_partidos_elo1['game_date'] < ultima_fecha]# - pd.Timedelta(days=30)]
partidos_elo1_test = df_partidos_elo1[df_partidos_elo1['game_date'] >= ultima_fecha]# - pd.Timedelta(days=30)]
data_entrada_elo1, data_entrada_elo1_test = escalar_datos(partidos_elo1, partidos_elo1_test, cols_equipos, cols_fecha + cols_elo)
data_salida_elo1, data_salida_elo1_test = np.array(partidos_elo1[['pts_home', 'pts_away']]), np.array(partidos_elo1_test[['pts_home', 'pts_away']])

#Partidos con ELO2
df_partidos_elo2 = pd.read_csv('csv_red/partidos_elo2.csv')
df_partidos_elo2, cols_equipos = preparar_datos_ohe(df_partidos_elo2, cols_equipos)

partidos_elo2 = df_partidos_elo2[df_partidos_elo2['game_date'] < ultima_fecha]# - pd.Timedelta(days=30)]
partidos_elo2_test = df_partidos_elo2[df_partidos_elo2['game_date'] >= ultima_fecha]# - pd.Timedelta(days=30)]
data_entrada_elo2, data_entrada_elo2_test = escalar_datos(partidos_elo2, partidos_elo2_test, cols_equipos, cols_fecha + cols_elo)
data_salida_elo2, data_salida_elo2_test = np.array(partidos_elo2[['pts_home', 'pts_away']]), np.array(partidos_elo2_test[['pts_home', 'pts_away']])

# Redes Neuronales

def crear_modelo(n_input):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=[n_input]),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2)
    ])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        #una poca cantidad de errores grandes es peor que muchos errores pequeños
        loss='mean_squared_error' 
    )

    return modelo

# Modelo base
modelo = crear_modelo(data_entrada.shape[1])
print("entrenando modelo...")
history = modelo.fit(data_entrada, data_salida, epochs=500, verbose=0)
print("modelo entrenado")
# Pérdida del modelo base
loss = modelo.evaluate(data_entrada, data_salida, verbose=0)
print(f'Pérdida del modelo base: {loss:.4f}')

# Modelo con ELO1
modelo_elo1 = crear_modelo(data_entrada_elo1.shape[1])
print("entrenando modelo con ELO1 ...")
history_elo1 = modelo_elo1.fit(data_entrada_elo1, data_salida_elo1, epochs=500, verbose=0)
print("modelo con ELO1 entrenado")
# Pérdida del modelo con ELO1
loss_elo1 = modelo_elo1.evaluate(data_entrada_elo1, data_salida_elo1, verbose=0)
print(f"Pérdida del modelo con ELO1: {loss_elo1:.4f}")

# Modelo con ELO2
modelo_elo2 = crear_modelo(data_entrada_elo2.shape[1])
print("entrenando modelo con ELO2 ...")
history_elo2 = modelo_elo2.fit(data_entrada_elo2, data_salida_elo2, epochs=500, verbose=0)
print("modelo con ELO2 entrenado")
# Pérdida del modelo con ELO2
loss_elo2 = modelo_elo2.evaluate(data_entrada_elo2, data_salida_elo2, verbose=0)
print(f"Pérdida del modelo con ELO2: {loss_elo2:.4f}")


# Evaluación de Modelos
from sklearn.metrics import mean_absolute_error

def evaluar_precision(modelo, entrada, salida, nombre_modelo):
    predicciones = modelo.predict(entrada)
    
    mae = mean_absolute_error(salida, predicciones) # (Error promedio en puntos)
    
    # Precision Exactitud del Ganador
    ganador_real = salida[:, 0] > salida[:, 1]
    ganador_pred = predicciones[:, 0] > predicciones[:, 1]
    
    precision = np.mean(ganador_real == ganador_pred) * 100
    
    print(f"\n--- Resultados {nombre_modelo} ---")
    print(f"Error Promedio (MAE): {mae:.2f} puntos")
    print(f"Precisión Ganador: {precision:.2f}%")
    return precision

# Evaluar los modelos
# Precisión de cada modelo con los partidos de la última fecha
acc_base = evaluar_precision(modelo, data_entrada_test, data_salida_test, "Modelo Base")
acc_elo1 = evaluar_precision(modelo_elo1, data_entrada_elo1_test, data_salida_elo1_test, "Modelo ELO1")
acc_elo2 = evaluar_precision(modelo_elo2, data_entrada_elo2_test, data_salida_elo2_test, "Modelo ELO2")