import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Evaluación de Modelos
from sklearn.metrics import mean_absolute_error
# Carpeta con los CSV de partidos
from elo2 import FOLDER

cols_fecha = ['year', 'month', 'day']
cols_elo = ['elo_h', 'elo_a', 'elo_h_ofg', 'elo_a_ofg', 'elo_h_dfg', 'elo_a_dfg', 
            'elo_h_ofg3', 'elo_a_ofg3', 'elo_h_dfg3', 'elo_a_dfg3'
            ]
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

def procesar_partidos(df, cols_cat, cols_num, corte_test=2017):
    # Filtrar partidos anteriores a 2018 para entrenamiento y posteriores para test
    temporada = df['season_id'].astype(str).str[-4:].astype(int)
    df_train = df[temporada <= corte_test]
    df_test = df[temporada > corte_test]
    
    data_entrada, data_entrada_test = escalar_datos(df_train, df_test, cols_cat, cols_num)
    
    data_salida = np.array(df_train[['pts_home', 'pts_away']])
    data_salida_test = np.array(df_test[['pts_home', 'pts_away']])
    
    return df_test, data_entrada, data_entrada_test, data_salida, data_salida_test


def cargar_procesar_elo(ruta_csv, cols_equipos_actuales, cols_cat, cols_num):
    df = pd.read_csv(ruta_csv)
    df, nuevas_cols_equipos = preparar_datos_ohe(df, cols_equipos_actuales)
    
    df, entrada_train, entrada_test, salida_train, salida_test = procesar_partidos(df, cols_cat, cols_num)
    
    return df, entrada_train, entrada_test, salida_train, salida_test, nuevas_cols_equipos


# Partidos Base
(partidos_test, data_entrada, data_entrada_test, data_salida, 
 data_salida_test, cols_equipos) = cargar_procesar_elo(FOLDER + 'partidos.csv', cols_equipos, [], cols_fecha)

# Partidos con Elo1
(partidos_elo1_test, data_entrada_elo1, data_entrada_elo1_test, data_salida_elo1, 
 data_salida_elo1_test, cols_equipos) = cargar_procesar_elo(FOLDER + 'partidos_elo1.csv', cols_equipos, [], cols_fecha + cols_elo)

# Partidos con Elo2
(partidos_elo2_test, data_entrada_elo2, data_entrada_elo2_test, data_salida_elo2, 
 data_salida_elo2_test, cols_equipos) = cargar_procesar_elo(FOLDER + 'partidos_elo2.csv', cols_equipos, [], cols_fecha + cols_elo)

# Partidos con Elo3
(partidos_elo3_test, data_entrada_elo3, data_entrada_elo3_test, data_salida_elo3, 
 data_salida_elo3_test, cols_equipos) = cargar_procesar_elo(FOLDER + 'partidos_elo3.csv', cols_equipos, ['playoffs'], cols_fecha + cols_elo)

# Red Neuronal

def crear_modelo(n_input):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=[n_input]),
        tf.keras.layers.Dropout(0.3), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2)
    ])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mean_squared_error' 
    )
    return modelo

# Evaluación de Modelos

def evaluar_precision(modelo, entrada, salida, nombre_modelo):
    predicciones = modelo.predict(entrada)
    predicciones = np.round(predicciones).astype(float)
        
    mae = mean_absolute_error(salida, predicciones) # (Error promedio en puntos)
        
    # Precision Exactitud del Ganador
    ganador_real = salida[:, 0] > salida[:, 1]
    ganador_pred = predicciones[:, 0] > predicciones[:, 1]
        
    precision = np.mean(ganador_real == ganador_pred) * 100
        
    print(f"\n--- Resultados {nombre_modelo} ---")
    print(f"Error Promedio (MAE): {mae:.2f} puntos")
    print(f"Precisión Ganador: {precision:.2f}%")
    return precision, mae

results_base = []
results_base_2 = []
results_elo1 = []
results_elo1_2 = []
results_elo2 = []
results_elo2_2 = []
results_elo3 = []
results_elo3_2 = []

#for i in range(20): #Descomentar y tabular el código de entrenamiento para realizar 20 iteraciones y obtener promedios de resultados

#callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)]-> Parar el entrenamiento sí la pérdida no mejora despúes de 20 épocas

# Modelo base
modelo = crear_modelo(data_entrada.shape[1])
print("entrenando modelo...")
history = modelo.fit(data_entrada, data_salida, epochs=2000, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)])
print("modelo entrenado")
# Pérdida del modelo base
loss = modelo.evaluate(data_entrada_test, data_salida_test, verbose=0)
print(f'Pérdida del modelo base: {loss:.4f}')

# Modelo con Elo1
modelo_elo1 = crear_modelo(data_entrada_elo1.shape[1])
print("entrenando modelo con Elo1 ...")
history_elo1 = modelo_elo1.fit(data_entrada_elo1, data_salida_elo1, epochs=2000, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)])
print("modelo con Elo1 entrenado")
loss_elo1 = modelo_elo1.evaluate(data_entrada_elo1_test, data_salida_elo1_test, verbose=0)
print(f"Pérdida del modelo con Elo1: {loss_elo1:.4f}")

# Modelo con Elo2
modelo_elo2 = crear_modelo(data_entrada_elo2.shape[1])
print("entrenando modelo con Elo2 ...")
history_elo2 = modelo_elo2.fit(data_entrada_elo2, data_salida_elo2, epochs=2000, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)])
print("modelo con Elo2 entrenado")
# Pérdida del modelo con Elo2
loss_elo2 = modelo_elo2.evaluate(data_entrada_elo2_test, data_salida_elo2_test, verbose=0)
print(f"Pérdida del modelo con Elo2: {loss_elo2:.4f}")

# Modelo con Elo3
modelo_elo3 = crear_modelo(data_entrada_elo3.shape[1])
print("entrenando modelo con Elo3 ...")
history_elo3 = modelo_elo3.fit(data_entrada_elo3, data_salida_elo3, epochs=2000, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)])
print("modelo con Elo3 entrenado")
# Pérdida del modelo con Elo3
loss_elo3 = modelo_elo3.evaluate(data_entrada_elo3_test, data_salida_elo3_test, verbose=0)
print(f"Pérdida del modelo con Elo3: {loss_elo3:.4f}")


# Evaluar los modelos
# Precisión de cada modelo con los partidos de la última fecha
acc_base, mae_base = evaluar_precision(modelo, data_entrada_test, data_salida_test, "Modelo Base")
acc_elo1, mae_elo1 = evaluar_precision(modelo_elo1, data_entrada_elo1_test, data_salida_elo1_test, "Modelo Elo1")
acc_elo2, mae_elo2 = evaluar_precision(modelo_elo2, data_entrada_elo2_test, data_salida_elo2_test, "Modelo Elo2")
acc_elo3, mae_elo3 = evaluar_precision(modelo_elo3, data_entrada_elo3_test, data_salida_elo3_test, "Modelo Elo3")

"""
# Descomentar para guardar resultados de todas las predicciones en CSV
def guardar_resultados_csv(df, modelo, entrada, nombre_archivo):
    df = df.copy()
    df = df[['season_id', 'game_date', 'team_name_home', 'team_name_away', 'pts_home', 'pts_away']]
    predicciones = modelo.predict(entrada)
    predicciones = np.round(predicciones).astype(float)
    df['pred_pts_home'] = predicciones[:, 0]
    df['pred_pts_away'] = predicciones[:, 1]
    df['delta_pts_home'] = df['pts_home'] - df['pred_pts_home']
    df['delta_pts_away'] = df['pts_away'] - df['pred_pts_away']
    df.to_csv('resultados/' + nombre_archivo, index=False)

guardar_resultados_csv(partidos_test, modelo, data_entrada_test, 'resultados_modelo_base.csv')
guardar_resultados_csv(partidos_elo1_test, modelo_elo1, data_entrada_elo1_test, 'resultados_modelo_elo1.csv')
guardar_resultados_csv(partidos_elo2_test, modelo_elo2, data_entrada_elo2_test, 'resultados_modelo_elo2.csv')
guardar_resultados_csv(partidos_elo3_test, modelo_elo3, data_entrada_elo3_test, 'resultados_modelo_elo3.csv')
"""

results_base.append((acc_base, mae_base))
results_elo1.append((acc_elo1, mae_elo1))
results_elo2.append((acc_elo2, mae_elo2))
results_elo3.append((acc_elo3, mae_elo3))
# Fin entrenamiento y evaluación de modelos

# Resultados Promedio
def calcular_promedios_y_mejor(nombre_modelo, results):
    avg_acc = np.mean([r[0] for r in results])
    avg_mae = np.mean([r[1] for r in results])
    best_acc = max(results, key=lambda x: x[0])
    best_mae = min(results, key=lambda x: x[1])

    print(f"\n--- Promedios {nombre_modelo} ---")
    print(f"Precisión Ganador: {avg_acc:.2f}%")
    print(f"Error Promedio (MAE): {avg_mae:.2f} puntos")
    print(f"\n--- Mejor {nombre_modelo} ---")
    print(f"Precisión Ganador: {best_acc[0]:.2f}%")
    print(f"Error Promedio (MAE): {best_mae[1]:.2f} puntos")

calcular_promedios_y_mejor("Modelo Base", results_base)
calcular_promedios_y_mejor("Modelo Elo1", results_elo1)
calcular_promedios_y_mejor("Modelo Elo2", results_elo2)
calcular_promedios_y_mejor("Modelo Elo3", results_elo3)