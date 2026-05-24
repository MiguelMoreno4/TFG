import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Cargar datos
summary = pd.read_csv("csv/game_summary.csv")
games = pd.read_csv("csv/game.csv")

# Unir por game_id
df = pd.merge(summary, games, on="game_id")

# Normalizar wl_home
df['wl_home'] = df['wl_home'].astype(str).str.strip().str.upper()

# Normalizar wl_away
df['wl_away'] = df['wl_away'].astype(str).str.strip().str.upper()

# Victoria del local a 1 y 0
df['win_home'] = df['wl_home'].apply(lambda x: 1 if x == 'W' else 0)

# Victoria como visitante a 1 y 0
df['win_away'] = df['wl_away'].apply(lambda x: 1 if x == 'W' else 0)

# Victorias locales totales por temporada
team_wins_home = df.groupby(['season', 'home_team_id'])['win_home'].sum().reset_index()
team_wins_home.rename(columns={'win_home': 'wins_home'}, inplace=True)

#Victorias visitantes totales por temporada
team_wins_away = df.groupby(['season', 'visitor_team_id'])['win_away'].sum().reset_index()
team_wins_away.rename(columns={'win_away': 'wins_away'}, inplace=True)

#Victorias totales (local y visitante) por temporada
team_total = pd.merge(team_wins_home, team_wins_away,
                      left_on=['season', 'home_team_id'],
                      right_on=['season', 'visitor_team_id'],
                      how='outer')

# Crear columna team_id unificando home_team_id y visitor_team_id
team_total['team_id'] = team_total['home_team_id'].combine_first(team_total['visitor_team_id'])

# Crear columna total_wins sumando wins_home + wins_away
team_total['total_wins'] = team_total['wins_home'].fillna(0) + team_total['wins_away'].fillna(0)

# Seleccionar solo columnas necesarias
team_total = team_total[['season', 'team_id', 'total_wins']]

# Introducimos el id del equipo y el año
team_id = 1610612744
year = 2023

team_data = team_total[team_total['team_id'] == team_id].sort_values('season')

print(f"\nVictorias por temporada del equipo {team_id}")
print("---------------------------------------------------")

for _, row in team_data.iterrows():
    print(f"Temporada {int(row['season'])}: {int(row['total_wins'])} victorias")



# Crear las 5 temporadas anteriores como features
team_total['season'] = team_total['season'].astype(int)
team_total = team_total.sort_values(['team_id', 'season'])

window = 5      #para usar 5 temporadas anteriores

#Vectores
X_list, y_list, team_ids, seasons_target = [], [], [], []

for team, group in team_total.groupby('team_id'):
    group = group.sort_values('season')
    wins = group['total_wins'].values
    seasons = group['season'].values

    # Crear todas las combinaciones de 5 temporadas consecutivas para predecir la siguiente
    for i in range(len(wins) - window):
        X_list.append(wins[i:i + window])
        y_list.append(wins[i + window])
        team_ids.append(team)
        seasons_target.append(seasons[i + window])

# Convertir a DataFrame
X = np.array(X_list)
y = np.array(y_list)
df_train = pd.DataFrame(X, columns=[f'wins_lag_{i}' for i in range(window, 0, -1)])
df_train['target_wins'] = y
df_train['team_id'] = team_ids
df_train['season_target'] = seasons_target

print(f"\nTotal de muestras creadas: {len(df_train)}")
print(df_train.head(10))

# Escalado y división de datos

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear y entrenar modelo

model = Sequential([
    Dense(64, activation='relu', input_shape=(window,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

model.fit(X_train, y_train, validation_split=0.1, epochs=150, batch_size=16, callbacks=[es], verbose=1)

# Evaluar
loss, mae = model.evaluate(X_test, y_test)
print(f"\nMAE en test: {mae:.2f}")


def predecir_equipo(team_id, year_to_predict=None):
    team_data = team_total[team_total['team_id'] == team_id].sort_values('season')
    wins = team_data['total_wins'].values
    seasons = team_data['season'].values

    if len(wins) <= window:
        print("No hay suficientes temporadas para generar predicciones.")
        return

    print(f"\nPredicciones para el equipo {team_id}")
    print("---------------------------------------------------")

    for i in range(len(wins) - window):
        prev_5 = wins[i:i+window].reshape(1, -1)
        target_season = seasons[i + window]

        prev_5_scaled = scaler.transform(prev_5)
        pred = model.predict(prev_5_scaled, verbose=0)

        # Mostrar siempre si no se filtra por año
        if year_to_predict is None or target_season == year_to_predict:
            real_val = wins[i+window] if i+window < len(wins) else "N/A"
            print(f"Temporadas {seasons[i]}–{seasons[i+window-1]} → "
                  f"predicción {target_season}: {pred[0][0]:.1f} victorias "
                  f"(real: {real_val})")

    if year_to_predict and year_to_predict not in seasons:
        # Predecir una temporada futura
        prev_seasons = team_data[team_data['season'] < year_to_predict].tail(window)
        if len(prev_seasons) == window:
            last5 = prev_seasons['total_wins'].values.reshape(1, -1)
            last5_scaled = scaler.transform(last5)
            pred_future = model.predict(last5_scaled, verbose=0)
            print(f"\nPredicción FUTURA para {year_to_predict}: {pred_future[0][0]:.1f} victorias")
        else:
            print(f"\nNo hay suficientes datos previos para predecir {year_to_predict}.")

predecir_equipo(team_id, year)