import pandas as pd
import numpy as np

def historial_5temporadas(df, fecha_objetivo):
    df['game_date'] = pd.to_datetime(df['game_date'])
    fecha_obj = pd.to_datetime(fecha_objetivo)
    
    # Encontrar la temporada del partido más cercano a la fecha
    temp_ref = int(str(df[df['game_date'] <= fecha_obj]['season_id'][-4:].max())[-4:])
    temp_inicio = temp_ref - 5
    
    mask = (df['season_id'].astype(str).str[-4:].astype(int) >= temp_inicio) & (df['game_date'] <= fecha_obj)
    return df[mask].sort_values('game_date')

def prob_esperada(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400)) # Fórmula estándar de Elo

def calcular_elos_fg(df_games, elo_ofensivo, elo_defensivo, i, partido, k):
    cols = ['fg_pct_home', 'fg_pct_away', 'fgm_home', 'fgm_away'] 
    df = df_games.copy().dropna(subset=cols) # Asegurarse de que no haya NaN en las columnas necesarias
    # Si no existen las columnas de ELO, inicializarlas
    for c in ['elo_h_ofg', 'elo_a_ofg', 'elo_h_dfg', 'elo_a_dfg']:
        if c not in df.columns:
            df[c] = 0.0
    
    id_h, id_a = partido['team_id_home'], partido['team_id_away']
    # Elo previo al partido
    df.at[i, 'elo_h_ofg'] = elo_ofensivo[id_h]
    df.at[i, 'elo_a_ofg'] = elo_ofensivo[id_a]
    df.at[i, 'elo_h_dfg'] = elo_defensivo[id_h]
    df.at[i, 'elo_a_dfg'] = elo_defensivo[id_a]
    
    # ELO OFENSIVO, EFICACIA: Quien tiene mejor porcentaje de tiros de campo anotados
    if partido['fg_pct_home'] > partido['fg_pct_away']:
        win_off_h = 1  
    elif partido['fg_pct_home'] < partido['fg_pct_away']:
        win_off_h = 0  
    else:
        win_off_h = 1 if partido['fgm_home'] > partido['fgm_away'] else 0
    
    # Si empatan en todo, se considera empate ofensivo
    if partido['fg_pct_home'] == partido['fg_pct_away'] and partido['fgm_home'] == partido['fgm_away']:
        win_off_h = 0.5

    exp_off_h = prob_esperada(elo_ofensivo[id_h], elo_ofensivo[id_a])
    ajuste_off = k * (win_off_h - exp_off_h)
    elo_ofensivo[id_h] += ajuste_off
    elo_ofensivo[id_a] -= ajuste_off

    # ELO DEFENSIVO, RESISTENCIA: Quien permite menos tiros de campo anotados
    if partido['fgm_away'] < partido['fgm_home']:
        win_def_h = 1 
    elif partido['fgm_away'] > partido['fgm_home']:
        win_def_h = 0 
    else:
        # Empate en triples metidos, decidimos por porcentaje permitido
        win_def_h = 1 if partido['fg_pct_away'] < partido['fg_pct_home'] else 0  
    # Si empatan en todo, se considera empate defensivo
    if partido['fgm_away'] == partido['fgm_home'] and partido['fg_pct_away'] == partido['fg_pct_home']:
        win_def_h = 0.5

    exp_def_h = prob_esperada(elo_defensivo[id_h], elo_defensivo[id_a])
    ajuste_def = k * (win_def_h - exp_def_h)
    elo_defensivo[id_h] += ajuste_def
    elo_defensivo[id_a] -= ajuste_def

    return df

def calcular_elos_fg3(df, elo_ofensivo, elo_defensivo, i, partido, k):
    # Si no existen las columnas de ELO, inicializarlas
    for c in ['elo_h_ofg3', 'elo_a_ofg3', 'elo_h_dfg3', 'elo_a_dfg3']:
        if c not in df.columns:
            df[c] = 0.0

    id_h, id_a = partido['team_id_home'], partido['team_id_away']
    # Elo previo al partido
    df.at[i, 'elo_h_ofg3'] = elo_ofensivo[id_h]
    df.at[i, 'elo_a_ofg3'] = elo_ofensivo[id_a]
    df.at[i, 'elo_h_dfg3'] = elo_defensivo[id_h]
    df.at[i, 'elo_a_dfg3'] = elo_defensivo[id_a]
    
    # ELO OFENSIVO, EFICACIA: Quien tiene mejor porcentaje de triples anotados
    if partido['fg3_pct_home'] > partido['fg3_pct_away']:
        win_off_h = 1 
    elif partido['fg3_pct_home'] < partido['fg3_pct_away']:
        win_off_h = 0 
    else:
        # Empate en porcentaje, decidimos por cantidad de triples metidos
        win_off_h = 1 if partido['fg3m_home'] > partido['fg3m_away'] else 0
    
    # Si empatan en todo, se considera empate ofensivo
    if partido['fg3_pct_home'] == partido['fg3_pct_away'] and partido['fg3m_home'] == partido['fg3m_away']:
        win_off_h = 0.5

    exp_off_h = prob_esperada(elo_ofensivo[id_h], elo_ofensivo[id_a])
    ajuste_off = k * (win_off_h - exp_off_h)
    elo_ofensivo[id_h] += ajuste_off
    elo_ofensivo[id_a] -= ajuste_off

    # ELO DEFENSIVO, RESISTENCIA: Quien permite menos triples anotados
    if partido['fg3m_away'] < partido['fg3m_home']:
        win_def_h = 1  
    elif partido['fg3m_away'] > partido['fg3m_home']:
        win_def_h = 0  
    else:
        # Empate en triples metidos, decidimos por porcentaje permitido
        win_def_h = 1 if partido['fg3_pct_away'] < partido['fg3_pct_home'] else 0
        
    # Si empatan en todo, se considera empate defensivo
    if partido['fg3m_away'] == partido['fg3m_home'] and partido['fg3_pct_away'] == partido['fg3_pct_home']:
        win_def_h = 0.5

    exp_def_h = prob_esperada(elo_defensivo[id_h], elo_defensivo[id_a])
    ajuste_def = k * (win_def_h - exp_def_h)
    elo_defensivo[id_h] += ajuste_def
    elo_defensivo[id_a] -= ajuste_def

    return df

def calcular_elo_nba(df_games, df_equipos, fecha, k=20): 
    cols = ['pts_home', 'pts_away', 'wl_home', 'team_id_home', 'team_id_away']
    df = historial_5temporadas(df_games, fecha).dropna(subset=cols) # Asegurarse de que no haya NaN en las columnas necesarias

    # Elo inicial 1500
    elos = dict.fromkeys(df_equipos['id'], 1500.0) 
    elo_ofensivo_fg = dict.fromkeys(df_equipos['id'], 1500.0)
    elo_defensivo_fg = dict.fromkeys(df_equipos['id'], 1500.0)
    elo_defensivo_fg3 = dict.fromkeys(df_equipos['id'], 1500.0)
    elo_ofensivo_fg3 = dict.fromkeys(df_equipos['id'], 1500.0)

    # Inicialización de columnas de ELO
    for c in ['elo_h', 'elo_a']:
        df[c] = 0.0

    for i, fila in df.iterrows():
        df = calcular_elos_fg(df, elo_ofensivo_fg, elo_defensivo_fg, i, fila, k)
        df = calcular_elos_fg3(df, elo_ofensivo_fg3, elo_defensivo_fg3, i, fila, k)

        id_h, id_a = fila['team_id_home'], fila['team_id_away']
        pts_h, pts_a = fila['pts_home'], fila['pts_away']
        
        # Elo previo al partido
        df.at[i, 'elo_h'] = elos[id_h]
        df.at[i, 'elo_a'] = elos[id_a]

        # Cálculo de puntos
        exp_home = prob_esperada(elos[id_h], elos[id_a])
        real_home = 1 if fila['wl_home'] == 'W' else 0

         # Factor de margen de victoria (FiveThirtyEight)
        mov = abs(pts_h - pts_a)
        multiplicador_mov = (mov + 3) ** 0.8 / (7.5 + 0.006 * mov)

        # Actualización
        puntos = k * (real_home - exp_home) * multiplicador_mov
        elos[id_h] += puntos
        elos[id_a] -= puntos  
        
    return df


def calcular_elo_nba_2(df_games, df_equipos, fecha, k=20): 
    cols = ['pts_home', 'pts_away', 'wl_home', 'team_id_home', 'team_id_away']
    df = historial_5temporadas(df_games, fecha).dropna(subset=cols) # Asegurarse de que no haya NaN en las columnas necesarias

    # Elo inicial 1500
    elos = dict.fromkeys(df_equipos['id'], 1500.0) 
    elo_ofensivo_fg = dict.fromkeys(df_equipos['id'], 1500.0)
    elo_defensivo_fg = dict.fromkeys(df_equipos['id'], 1500.0)
    elo_defensivo_fg3 = dict.fromkeys(df_equipos['id'], 1500.0)
    elo_ofensivo_fg3 = dict.fromkeys(df_equipos['id'], 1500.0)
    
    # Reseteo de Elo al inicio de cada temporada(FiveThirtyEight)
    def reset_elo_temporada(elos):
        for equipo_id in elos.keys():
            elos[equipo_id] = elos[equipo_id] * 0.75 + 1500 * 0.25 
        return elos

    # Valor de season_id en la primera fila
    temporada_actual = df.iloc[0]['season_id']

    # Inicialización de columnas de ELO
    for c in ['elo_h', 'elo_a']:
        df[c] = 0.0

    for i, fila in df.iterrows():
        df = calcular_elos_fg(df, elo_ofensivo_fg, elo_defensivo_fg, i, fila, k)
        df = calcular_elos_fg3(df, elo_ofensivo_fg3, elo_defensivo_fg3, i, fila, k)

        id_h, id_a = fila['team_id_home'], fila['team_id_away']
        pts_h, pts_a = fila['pts_home'], fila['pts_away']
        
        # Reseteo de Elo al inicio de cada temporada
        if fila['season_id'] != temporada_actual:
            elos = reset_elo_temporada(elos)
            temporada_actual = fila['season_id']

        # Elo previo al partido
        df.at[i, 'elo_h'] = elos[id_h]
        df.at[i, 'elo_a'] = elos[id_a]

        # Cálculo de puntos
        exp_home = prob_esperada(elos[id_h] + 100, elos[id_a]) # Ventaja de 100 puntos Elo para el equipo local
        real_home = 1 if fila['wl_home'] == 'W' else 0
        
         # Factor de margen de victoria (FiveThirtyEight)
        mov = abs(pts_h - pts_a)
        multiplicador_mov = (mov + 3) ** 0.8 / (7.5 + 0.006 * mov)

        # Actualización
        puntos = k * (real_home - exp_home) * multiplicador_mov
        elos[id_h] += puntos
        elos[id_a] -= puntos  
    #sustituir 
    return df

df_partidos = pd.read_csv('csv/game.csv')
df_equipos = pd.read_csv('csv/team.csv')

# Descartar partidos que no sean Regular Season o Playoffs
df_partidos = df_partidos[df_partidos['season_type'].isin(['Regular Season', 'Playoffs'])]
df_partidos['game_date'] = pd.to_datetime(df_partidos['game_date'])
# seleccionar columnas relevantes sin duplicados (evita problemas al iterar con iterrows)
df_partidos = df_partidos[[
    'game_date',
    'team_id_home',
    'team_id_away',
    'team_name_home',
    'team_name_away',
    'team_abbreviation_home',
    'team_abbreviation_away',
    'pts_home',
    'pts_away',
    'wl_home',
    'season_id',
    'fg_pct_home',
    'fg_pct_away',
    'fgm_home',
    'fgm_away',
    'fg3_pct_home',
    'fg3_pct_away',
    'fg3m_home',
    'fg3m_away'
]]

fecha = '2019-04-05' # 13 partidos esa fecha para testeo en v1.py
# Obtener los ELOs justo antes de esa fecha
#ELO 1
df_partidos_elo1 = calcular_elo_nba(df_partidos, df_equipos, fecha)

#ELO 2
df_partidos_elo2 = calcular_elo_nba_2(df_partidos, df_equipos, fecha)

# Mostrar las últimas 5 filas con los nuevos ELOs calculados
print("ELO Modelo 1:")
#print partidos sin indexar y sin team_id
print(df_partidos_elo1.drop(columns=['team_id_home', 'team_id_away', 'pts_home', 'pts_away', 'wl_home','season_id','fg_pct_home',
    'fg_pct_away',
    'fgm_home',
    'fgm_away',
    'fg3_pct_home',
    'fg3_pct_away',
    'fg3m_home',
    'fg3m_away']).tail(5))
print("\nELO Modelo 2:")
print(df_partidos_elo2.drop(columns=['team_id_home', 'team_id_away', 'pts_home', 'pts_away', 'wl_home','season_id', 'fg_pct_home',
    'fg_pct_away',
    'fgm_home',
    'fgm_away',
    'fg3_pct_home',
    'fg3_pct_away',
    'fg3m_home',
    'fg3m_away']).tail(5))

cols = ['team_id_home', 'team_id_away','team_name_home','team_name_away','team_abbreviation_home','team_abbreviation_away', 'game_date', 'pts_home', 'pts_away']
cols2 = ['elo_h', 'elo_a', 'elo_h_ofg', 'elo_a_ofg', 'elo_h_dfg', 'elo_a_dfg', 'elo_h_ofg3', 'elo_a_ofg3', 'elo_h_dfg3', 'elo_a_dfg3']

df_partidos = historial_5temporadas(df_partidos, fecha)[cols]
df_partidos_elo1 = df_partidos_elo1[cols + cols2]
df_partidos_elo2 = df_partidos_elo2[cols + cols2]

# Guardar los 3 DataFrames en archivos CSV
df_partidos.to_csv('csv_red/partidos.csv', index=False)
df_partidos_elo1.to_csv('csv_red/partidos_elo1.csv', index=False)
df_partidos_elo2.to_csv('csv_red/partidos_elo2.csv', index=False)
