import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Evaluación de Modelos
from sklearn.metrics import mean_absolute_error

cols_fecha = ['year', 'month', 'day']
cols_elo = ['elo_h', 'elo_a']#, 'elo_h_ofg', 'elo_a_ofg', 'elo_h_dfg', 'elo_a_dfg', 
            #'elo_h_ofg3', 'elo_a_ofg3', 'elo_h_dfg3', 'elo_a_dfg3']
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
def escalar_datos(df, df_test, cols_no_escalables, cols_escalables):
    scaler = StandardScaler()
   
    # Ajustar escalador solo con las columnas fecha y ELO de los datos de entrenamiento
    scaler.fit(df[cols_escalables])

    # Transformar sets
    df_escalado = scaler.transform(df[cols_escalables])
    df_test_escalado = scaler.transform(df_test[cols_escalables])
    # Combinar con las columnas no escalables
    data_entrada = np.hstack([np.array(df[cols_no_escalables]), df_escalado])
    data_entrada_test = np.hstack([np.array(df_test[cols_no_escalables]), df_test_escalado])
    
    return data_entrada, data_entrada_test

def preparar_datos_salida(df):
    home_win = (df['pts_home'] > df['pts_away']).astype(int)
    away_win = (df['pts_away'] > df['pts_home']).astype(int)

    return np.column_stack((home_win, away_win))

# Partidos con datos básicos
df_partidos = pd.read_csv('csv_red/partidos.csv')
ultima_fecha = pd.to_datetime(df_partidos['game_date'].max())
df_partidos, cols_equipos = preparar_datos_ohe(df_partidos, cols_equipos)

# Filtrar partidos anteriores al último mes de datos para entrenamiento
partidos = df_partidos[df_partidos['season_id'].astype(str).str[-4:].astype(int) <= 2017] #[df_partidos['game_date'] < ultima_fecha - pd.Timedelta(days=30)]
partidos_test = df_partidos[df_partidos['season_id'].astype(str).str[-4:].astype(int) > 2017] #[df_partidos['game_date'] >= ultima_fecha - pd.Timedelta(days=30)]
data_entrada, data_entrada_test = escalar_datos(partidos, partidos_test, cols_equipos, cols_fecha)
data_salida, data_salida_test = preparar_datos_salida(partidos), preparar_datos_salida(partidos_test)

#Partidos con ELO1
df_partidos_elo1 = pd.read_csv('csv_red/partidos_elo1.csv')
df_partidos_elo1, cols_equipos = preparar_datos_ohe(df_partidos_elo1, cols_equipos)

partidos_elo1 = df_partidos_elo1[df_partidos_elo1['season_id'].astype(str).str[-4:].astype(int) <= 2017] #[df_partidos_elo1['game_date'] < ultima_fecha - pd.Timedelta(days=30)]
partidos_elo1_test = df_partidos_elo1[df_partidos_elo1['season_id'].astype(str).str[-4:].astype(int) > 2017] #[df_partidos_elo1['game_date'] >= ultima_fecha - pd.Timedelta(days=30)]
data_entrada_elo1, data_entrada_elo1_test = escalar_datos(partidos_elo1, partidos_elo1_test, [], cols_fecha + cols_elo)
data_salida_elo1, data_salida_elo1_test = preparar_datos_salida(partidos_elo1), preparar_datos_salida(partidos_elo1_test)

#Partidos con ELO2
df_partidos_elo2 = pd.read_csv('csv_red/partidos_elo2.csv')
df_partidos_elo2, cols_equipos = preparar_datos_ohe(df_partidos_elo2, cols_equipos)

partidos_elo2 = df_partidos_elo2[df_partidos_elo2['season_id'].astype(str).str[-4:].astype(int) <= 2017] #[df_partidos_elo2['game_date'] < ultima_fecha - pd.Timedelta(days=30)]
partidos_elo2_test = df_partidos_elo2[df_partidos_elo2['season_id'].astype(str).str[-4:].astype(int) > 2017] #[df_partidos_elo2['game_date'] >= ultima_fecha - pd.Timedelta(days=30)]
data_entrada_elo2, data_entrada_elo2_test = escalar_datos(partidos_elo2, partidos_elo2_test, [], cols_fecha + cols_elo)
data_salida_elo2, data_salida_elo2_test = preparar_datos_salida(partidos_elo2), preparar_datos_salida(partidos_elo2_test)

# Partidos con ELO3
df_partidos_elo3 = pd.read_csv('csv_red/partidos_elo3.csv')
df_partidos_elo3, cols_equipos = preparar_datos_ohe(df_partidos_elo3, cols_equipos)
partidos_elo3 = df_partidos_elo3[df_partidos_elo3['season_id'].astype(str).str[-4:].astype(int) <= 2017] #[df_partidos_elo3['game_date'] < ultima_fecha - pd.Timedelta(days=30)]
partidos_elo3_test = df_partidos_elo3[df_partidos_elo3['season_id'].astype(str).str[-4:].astype(int) > 2017] #[df_partidos_elo3['game_date'] >= ultima_fecha - pd.Timedelta(days=30)]
data_entrada_elo3, data_entrada_elo3_test = escalar_datos(partidos_elo3, partidos_elo3_test,['playoffs'], cols_fecha + cols_elo)
data_salida_elo3, data_salida_elo3_test = preparar_datos_salida(partidos_elo3), preparar_datos_salida(partidos_elo3_test)

# Redes Neuronales

def crear_modelo(n_input):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=[n_input]),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation= 'softmax')
    ])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        #una poca cantidad de errores grandes es peor que muchos errores pequeños
        loss='mean_squared_error' 
    )

    return modelo

def crear_modelo_2(n_input):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=[n_input]),
        tf.keras.layers.Dropout(0.3), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation= 'softmax')
    ])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss='mean_squared_error' 
    )
    return modelo

# Evaluación de Modelos

def evaluar_precision(modelo, entrada, salida, nombre_modelo):
    predicciones = modelo.predict(entrada)
    predicciones = np.round(predicciones).astype(float)
        
    ganador_pred = np.argmax(predicciones, axis=1)
    ganador_real = np.argmax(salida, axis=1)

    mae = mean_absolute_error(ganador_real, ganador_pred)
    precision = np.mean(ganador_pred == ganador_real) * 100

    print(f"\n--- Resultados {nombre_modelo} ---")
    print(f"Error Promedio (MAE): {mae:.2f}")
    print(f"Precisión Ganador: {precision:.2f}%")

    return precision, mae

results_base = []
results_elo1 = []
results_elo2 = []
results_elo3 = []

# Parar el entrenamiento sí la pérdida no mejora despúes de 20 épocas

### for i in range(20):

# Modelo base
modelo = crear_modelo(data_entrada.shape[1])
modelo_2 = crear_modelo_2(data_entrada.shape[1])
print("entrenando modelo...")
history = modelo.fit(data_entrada, data_salida, epochs=2000, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)])
history_2 = modelo_2.fit(data_entrada, data_salida, epochs=2000, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)])
print("modelo entrenado")
# Pérdida del modelo base
loss = modelo.evaluate(data_entrada_test, data_salida_test, verbose=0)
loss_2 = modelo_2.evaluate(data_entrada_test, data_salida_test, verbose=0)
print(f'Pérdida del modelo base: {loss:.4f}')
print(f'Pérdida del modelo base_2: {loss_2:.4f}')

# Modelo con ELO1
modelo_elo1 = crear_modelo(data_entrada_elo1.shape[1])
modelo_elo1_2 = crear_modelo_2(data_entrada_elo1.shape[1])
print("entrenando modelo con ELO1 ...")
history_elo1 = modelo_elo1.fit(data_entrada_elo1, data_salida_elo1, epochs=2000, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)])
history_elo1_2 = modelo_elo1_2.fit(data_entrada_elo1, data_salida_elo1, epochs=2000, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)])
print("modelo con ELO1 entrenado")
# Pérdida del modelo con ELO1
loss_elo1 = modelo_elo1.evaluate(data_entrada_elo1_test, data_salida_elo1_test, verbose=0)
loss_elo1_2 = modelo_elo1_2.evaluate(data_entrada_elo1_test, data_salida_elo1_test, verbose=0)
print(f"Pérdida del modelo con ELO1: {loss_elo1:.4f}")
print(f"Pérdida del modelo con ELO1_2: {loss_elo1_2:.4f}")

# Modelo con ELO2
modelo_elo2 = crear_modelo(data_entrada_elo2.shape[1])
modelo_elo2_2 = crear_modelo_2(data_entrada_elo2.shape[1])
print("entrenando modelo con ELO2 ...")
history_elo2 = modelo_elo2.fit(data_entrada_elo2, data_salida_elo2, epochs=2000, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)])
history_elo2_2 = modelo_elo2_2.fit(data_entrada_elo2, data_salida_elo2, epochs=2000, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)])
print("modelo con ELO2 entrenado")
# Pérdida del modelo con ELO2
loss_elo2 = modelo_elo2.evaluate(data_entrada_elo2_test, data_salida_elo2_test, verbose=0)
loss_elo2_2 = modelo_elo2_2.evaluate(data_entrada_elo2_test, data_salida_elo2_test, verbose=0)
print(f"Pérdida del modelo con ELO2: {loss_elo2:.4f}")
print(f"Pérdida del modelo con ELO2_2: {loss_elo2_2:.4f}")

# Modelo con ELO3
modelo_elo3 = crear_modelo(data_entrada_elo3.shape[1])
modelo_elo3_2 = crear_modelo_2(data_entrada_elo3.shape[1])
print("entrenando modelo con ELO3 ...")
history_elo3 = modelo_elo3.fit(data_entrada_elo3, data_salida_elo3, epochs=2000, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)])
history_elo3_2 = modelo_elo3_2.fit(data_entrada_elo3, data_salida_elo3, epochs=2000, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)])
print("modelo con ELO3 entrenado")
# Pérdida del modelo con ELO3
loss_elo3 = modelo_elo3.evaluate(data_entrada_elo3_test, data_salida_elo3_test, verbose=0)
loss_elo3_2 = modelo_elo3_2.evaluate(data_entrada_elo3_test, data_salida_elo3_test, verbose=0)
print(f"Pérdida del modelo con ELO3: {loss_elo3:.4f}")
print(f"Pérdida del modelo con ELO3_2: {loss_elo3_2:.4f}")

# Evaluar los modelos
# Precisión de cada modelo con los partidos de la última fecha
acc_base, mae_base = evaluar_precision(modelo, data_entrada_test, data_salida_test, "Modelo Base")
acc_base_2, mae_base_2 = evaluar_precision(modelo_2, data_entrada_test, data_salida_test, "Modelo Base_2")
acc_elo1, mae_elo1 = evaluar_precision(modelo_elo1, data_entrada_elo1_test, data_salida_elo1_test, "Modelo ELO1")
acc_elo1_2, mae_elo1_2 = evaluar_precision(modelo_elo1_2, data_entrada_elo1_test, data_salida_elo1_test, "Modelo ELO1_2")
acc_elo2, mae_elo2 = evaluar_precision(modelo_elo2, data_entrada_elo2_test, data_salida_elo2_test, "Modelo ELO2")
acc_elo2_2, mae_elo2_2 = evaluar_precision(modelo_elo2_2, data_entrada_elo2_test, data_salida_elo2_test, "Modelo ELO2_2")
acc_elo3, mae_elo3 = evaluar_precision(modelo_elo3, data_entrada_elo3_test, data_salida_elo3_test, "Modelo ELO3")
acc_elo3_2, mae_elo3_2 = evaluar_precision(modelo_elo3_2, data_entrada_elo3_test, data_salida_elo3_test, "Modelo ELO3_2")


# Guardar resultados en CSV
def guardar_resultados_csv(df, modelo, entrada, nombre_archivo):
    df = df.copy()
    df = df_partidos[df_partidos['season_id'].astype(str).str[-4:].astype(int) > 2017]
    df = df[['season_id', 'game_date', 'team_name_home', 'team_name_away', 'pts_home', 'pts_away']]
    df['home_win'] = df['pts_home'] > df['pts_away']
    predicciones = modelo.predict(entrada)
    df['pred_home_win'] = predicciones[:, 0] > predicciones[:, 1]
    df.to_csv('resultados2/' + nombre_archivo, index=False)

guardar_resultados_csv(partidos_test, modelo, data_entrada_test, 'resultados_modelo_base.csv')
guardar_resultados_csv(partidos_test, modelo_2, data_entrada_test, 'resultados_modelo_base_2.csv')
guardar_resultados_csv(partidos_elo1_test, modelo_elo1, data_entrada_elo1_test, 'resultados_modelo_elo1.csv')
guardar_resultados_csv(partidos_elo1_test, modelo_elo1_2, data_entrada_elo1_test, 'resultados_modelo_elo1_2.csv')
guardar_resultados_csv(partidos_elo2_test, modelo_elo2, data_entrada_elo2_test, 'resultados_modelo_elo2.csv')
guardar_resultados_csv(partidos_elo2_test, modelo_elo2_2, data_entrada_elo2_test, 'resultados_modelo_elo2_2.csv')
guardar_resultados_csv(partidos_elo3_test, modelo_elo3, data_entrada_elo3_test, 'resultados_modelo_elo3.csv')
guardar_resultados_csv(partidos_elo3_test, modelo_elo3_2, data_entrada_elo3_test, 'resultados_modelo_elo3_2.csv')

"""
    results_base.append((acc_base, mae_base))
    results_elo1.append((acc_elo1, mae_elo1))
    results_elo2.append((acc_elo2, mae_elo2))
    results_elo3.append((acc_elo3, mae_elo3))

# Resultados Promedio
def calcular_promedios_y_mejor(nombre_modelo, results):
    avg_acc = np.mean([r[0] for r in results])
    avg_mae = np.mean([r[1] for r in results])
    best_acc = max(results, key=lambda x: x[0])
    best_mae = min(results, key=lambda x: x[1])

    print(f"\n--- Promedios {nombre_modelo} ---")
    print(f"Precisión Ganador: {avg_acc:.2f}%")
    print(f"Error Promedio (MAE): {avg_mae:.2f} puntos")
    print(f"Mejor Precisión: {best_acc[0]:.2f}%")
    print(f"Mejor Error (MAE): {best_mae[1]:.2f} puntos")

calcular_promedios_y_mejor("Modelo Base", results_base)
calcular_promedios_y_mejor("Modelo ELO1", results_elo1)
calcular_promedios_y_mejor("Modelo ELO2", results_elo2)
calcular_promedios_y_mejor("Modelo ELO3", results_elo3)
"""