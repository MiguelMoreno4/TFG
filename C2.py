import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

# CARGA
df = pd.read_csv('csv_red/partidos_elo3.csv')
df = df.sort_values('game_date').reset_index(drop=True)
df['game_date'] = pd.to_datetime(df['game_date'])

# FEATURES NUEVAS: FORMA RECIENTE Y DESCANSO
# Para cada equipo calculamos sus últimos N resultados y puntos
# trabajando partido a partido en orden temporal

records = []  # aquí acumulamos el historial por equipo

# Diccionario: team_id -> lista de sus partidos anteriores
history = {}

for _, row in df.iterrows():
    h, a = row['team_id_home'], row['team_id_away']

    def get_stats(team_id, current_date):
        games = history.get(team_id, [])
        if len(games) == 0:
            return {
                'win_rate_5': 0.5, 'win_rate_10': 0.5,
                'avg_pts_5': 110.0, 'avg_pts_10': 110.0,
                'avg_pts_against_5': 110.0, 'avg_pts_against_10': 110.0,
                'rest_days': 3.0
            }
        last5  = games[-5:]
        last10 = games[-10:]
        last_date = games[-1]['date']
        rest = (current_date - last_date).days

        return {
            'win_rate_5':        np.mean([g['win'] for g in last5]),
            'win_rate_10':       np.mean([g['win'] for g in last10]),
            'avg_pts_5':         np.mean([g['pts_for'] for g in last5]),
            'avg_pts_10':        np.mean([g['pts_for'] for g in last10]),
            'avg_pts_against_5': np.mean([g['pts_against'] for g in last5]),
            'avg_pts_against_10':np.mean([g['pts_against'] for g in last10]),
            'rest_days':         min(rest, 7)
        }

    h_stats = get_stats(h, row['game_date'])
    a_stats = get_stats(a, row['game_date'])

    record = {
        # ELO diferencias
        'elo_diff':       row['elo_h']      - row['elo_a'],
        'elo_off_diff':   row['elo_h_ofg']  - row['elo_a_ofg'],
        'elo_def_diff':   row['elo_h_dfg']  - row['elo_a_dfg'],
        'elo_off3_diff':  row['elo_h_ofg3'] - row['elo_a_ofg3'],
        'elo_def3_diff':  row['elo_h_dfg3'] - row['elo_a_dfg3'],
        'is_playoffs':    int(row['playoffs']),

        # Forma reciente HOME
        'h_win_rate_5':         h_stats['win_rate_5'],
        'h_win_rate_10':        h_stats['win_rate_10'],
        'h_avg_pts_5':          h_stats['avg_pts_5'],
        'h_avg_pts_10':         h_stats['avg_pts_10'],
        'h_avg_pts_against_5':  h_stats['avg_pts_against_5'],
        'h_avg_pts_against_10': h_stats['avg_pts_against_10'],
        'h_rest_days':          h_stats['rest_days'],

        # Forma reciente AWAY
        'a_win_rate_5':         a_stats['win_rate_5'],
        'a_win_rate_10':        a_stats['win_rate_10'],
        'a_avg_pts_5':          a_stats['avg_pts_5'],
        'a_avg_pts_10':         a_stats['avg_pts_10'],
        'a_avg_pts_against_5':  a_stats['avg_pts_against_5'],
        'a_avg_pts_against_10': a_stats['avg_pts_against_10'],
        'a_rest_days':          a_stats['rest_days'],

        # Diferencias de forma
        'win_rate_diff_5':       h_stats['win_rate_5']  - a_stats['win_rate_5'],
        'win_rate_diff_10':      h_stats['win_rate_10'] - a_stats['win_rate_10'],
        'pts_diff_5':            h_stats['avg_pts_5']   - a_stats['avg_pts_5'],
        'rest_diff':             h_stats['rest_days']   - a_stats['rest_days'],

        # Objetivos
        'home_win': int(row['pts_home'] > row['pts_away']),
        'pts_home': row['pts_home'],
        'pts_away': row['pts_away'],
    }
    records.append(record)

    # Actualizar historial después de procesar
    home_win = row['pts_home'] > row['pts_away']
    if h not in history: history[h] = []
    if a not in history: history[a] = []

    history[h].append({
        'date': row['game_date'], 'win': int(home_win),
        'pts_for': row['pts_home'], 'pts_against': row['pts_away']
    })
    history[a].append({
        'date': row['game_date'], 'win': int(not home_win),
        'pts_for': row['pts_away'], 'pts_against': row['pts_home']
    })

df_feat = pd.DataFrame(records)

# SPLIT TEMPORAL
# Descartamos los primeros 10 partidos de cada equipo
df_feat = df_feat.iloc[int(len(df_feat) * 0.02):].reset_index(drop=True)

cutoff = int(len(df_feat) * 0.8)
train  = df_feat.iloc[:cutoff]
test   = df_feat.iloc[cutoff:]

feature_cols = [c for c in df_feat.columns if c not in ('home_win', 'pts_home', 'pts_away')]

X_train = train[feature_cols].values
X_test  = test[feature_cols].values
y_win_train = train['home_win'].values
y_pts_train = train[['pts_home', 'pts_away']].values
y_win_test  = test['home_win'].values
y_pts_test  = test[['pts_home', 'pts_away']].values

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# MODELO
inputs = tf.keras.Input(shape=(len(feature_cols),))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)

out_win = tf.keras.layers.Dense(1, activation='sigmoid', name='winner')(x)
out_pts = tf.keras.layers.Dense(2, activation='linear',  name='points')(x)

model = tf.keras.Model(inputs=inputs, outputs=[out_win, out_pts])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={'winner': 'binary_crossentropy', 'points': 'mae'},
    loss_weights={'winner': 1.0, 'points': 0.1},
    metrics={'winner': 'accuracy'}
)

model.fit(
    X_train,
    {'winner': y_win_train, 'points': y_pts_train},
    validation_data=(X_test, {'winner': y_win_test, 'points': y_pts_test}),
    epochs=150,
    batch_size=64,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
    verbose=1
)

# EVALUACIÓN
pred_win, pred_pts = model.predict(X_test)
pred_win_binary = (pred_win > 0.5).astype(int).flatten()

acc = accuracy_score(y_win_test, pred_win_binary)
mae = mean_absolute_error(y_pts_test, pred_pts)

print(f"\nAccuracy winner : {acc*100:.1f}%")
print(f"MAE puntos      : {mae:.4f}")