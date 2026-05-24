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

# Convertir game_date a datetime y definir season correctamente
df['game_date'] = pd.to_datetime(df['game_date_est'])
df['season'] = df['game_date'].dt.year
df.loc[df['game_date'].dt.month >= 10, 'season'] += 1  # Temporada termina en junio siguiente

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

# Seleccionar solo columnas necesarias y calcular partidos jugados
team_total['games_played'] = 0
for idx, row in team_total.iterrows():
    season = row['season']
    team_id_val = row['team_id']
    games_home = df[(df['season'] == season) & (df['home_team_id'] == team_id_val)].shape[0]
    games_away = df[(df['season'] == season) & (df['visitor_team_id'] == team_id_val)].shape[0]
    team_total.at[idx, 'games_played'] = games_home + games_away

# Calcular porcentaje de victorias
team_total['win_pct'] = team_total['total_wins'] / team_total['games_played']

# Filtrar temporadas con menos de 20 partidos
team_total = team_total[team_total['games_played'] >= 20]

# Solo dejar columnas necesarias
team_total = team_total[['season', 'team_id', 'total_wins', 'games_played', 'win_pct']]

# Equipo y temporada
team_id = 1610612744
year_to_predict = 2010


team_data = team_total[team_total['team_id'] == team_id].sort_values('season')

print(f"\nVictorias y porcentaje por temporada del equipo {team_id}")
print("---------------------------------------------------")
for _, row in team_data.iterrows():
    season = int(row['season'])
    total_wins = int(row['total_wins'])
    total_games = int(row['games_played'])
    win_pct = row['win_pct']
    print(f"Temporada {season}: {total_wins} victorias en {total_games} partidos ({win_pct:.3f} win%)")




# Crear las 5 temporadas anteriores como features (usando win_pct)
team_total['season'] = team_total['season'].astype(int)
team_total = team_total.sort_values(['team_id', 'season'])

window = 5      #para usar 5 temporadas anteriores

# Vectores
X_list, y_list, team_ids, seasons_target = [], [], [], []

for team, group in team_total.groupby('team_id'):
    group = group.sort_values('season')
    win_pcts = group['win_pct'].values
    seasons = group['season'].values

    # Crear todas las combinaciones de 5 temporadas consecutivas para predecir la siguiente
    for i in range(len(win_pcts) - window):
        X_list.append(win_pcts[i:i + window])
        y_list.append(win_pcts[i + window])
        team_ids.append(team)
        seasons_target.append(seasons[i + window])

# Convertir a DataFrame
X = np.array(X_list)
y = np.array(y_list)
df_train = pd.DataFrame(X, columns=[f'win_pct_lag_{i}' for i in range(window, 0, -1)])
df_train['target_win_pct'] = y
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
    tf.keras.Input(shape=(window,)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

model.fit(X_train, y_train, validation_split=0.1, epochs=150, batch_size=16, callbacks=[es], verbose=1)

# Evaluar
loss, mae = model.evaluate(X_test, y_test)
mae_victorias = mae * 82
print(f"\nMAE en test (porcentaje): {mae:.3f}")
print(f"MAE en test (victorias/82 partidos): {mae_victorias:.2f}")



def predecir_equipo(team_id, year_to_predict):
    # Filtrar datos hasta el año anterior al que queremos predecir
    team_total_filtered = team_total[team_total['season'] < year_to_predict]
    
    if team_total_filtered.empty:
        print(f"No hay datos suficientes antes de {year_to_predict}.")
        return
    
    # Recrear las 5 temporadas anteriores como features con datos filtrados (usando win_pct)
    team_total_filtered = team_total_filtered.sort_values(['team_id', 'season'])
    X_list, y_list, team_ids, seasons_target = [], [], [], []
    for team, group in team_total_filtered.groupby('team_id'):
        group = group.sort_values('season')
        win_pcts = group['win_pct'].values
        seasons = group['season'].values
        for i in range(len(win_pcts) - window):
            X_list.append(win_pcts[i:i + window])
            y_list.append(win_pcts[i + window])
            team_ids.append(team)
            seasons_target.append(seasons[i + window])
    if not X_list:
        print(f"No hay suficientes muestras para entrenar antes de {year_to_predict}.")
        return
    X = np.array(X_list)
    y = np.array(y_list)
    # Escalado y división
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # Entrenar modelo
    model = Sequential([
        tf.keras.Input(shape=(window,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.1, epochs=150, batch_size=16, callbacks=[es], verbose=0)
    # Evaluar
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    mae_victorias = mae * 82
    print(f"MAE en test para predicción de {year_to_predict} (porcentaje): {mae:.3f}")
    print(f"MAE en test para predicción de {year_to_predict} (victorias/82 partidos): {mae_victorias:.2f}")
    # Predecir para el equipo específico
    team_data = team_total_filtered[team_total_filtered['team_id'] == team_id].sort_values('season')
    win_pcts = team_data['win_pct'].values
    seasons = team_data['season'].values
    if len(win_pcts) < window:
        print("No hay suficientes temporadas para generar predicciones.")
        return
    print(f"\nPredicciones para el equipo {team_id} en {year_to_predict}")
    print("---------------------------------------------------")
    # Usar las últimas 5 temporadas disponibles para predecir el año objetivo
    last5 = win_pcts[-window:] if len(win_pcts) >= window else win_pcts
    if len(last5) == window:
        last5_scaled = scaler.transform(last5.reshape(1, -1))
        pred_pct = model.predict(last5_scaled, verbose=0)[0][0]
        pred_wins = pred_pct * 82
        print(f"Predicción para {year_to_predict}: {pred_pct:.3f} win% aprox. {pred_wins:.1f} victorias (en 82 partidos)")
    else:
        print(f"No hay suficientes datos para predecir {year_to_predict}.")

predecir_equipo(team_id, year_to_predict)