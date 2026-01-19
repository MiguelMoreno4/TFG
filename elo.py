import pandas as pd
import numpy as np

def historial_5temporadas(df, fecha_objetivo):
    df['game_date'] = pd.to_datetime(df['game_date'])
    fecha_obj = pd.to_datetime(fecha_objetivo)
    
    # Encontrar la temporada del partido más cercano a la fecha
    temp_ref = int(str(df[df['game_date'] <= fecha_obj]['season_id'].max())[-4:])
    temp_inicio = temp_ref - 5
    
    mask = (df['season_id'].astype(str).str[-4:].astype(int) >= temp_inicio) & (df['game_date'] < fecha_obj)
    return df[mask].sort_values('game_date')

def prob_esperada(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400)) # Fórmula estándar de Elo

def calcular_elo_nba(df_games, df_equipos, fecha, k=20): 
    cols = ['pts_home', 'pts_away', 'wl_home', 'team_id_home', 'team_id_away']
    df = historial_5temporadas(df_games, fecha).dropna(subset=cols) # Asegurarse de que no haya NaN en las columnas necesarias
    elos = dict.fromkeys(df_equipos['id'], 1500.0) # Elo inicial 1500

    for i, fila in df.iterrows():
        id_h, id_a = fila['team_id_home'], fila['team_id_away']
        pts_h, pts_a = fila['pts_home'], fila['pts_away']
        
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
        
    return elos


def calcular_elo_nba_2(df_games, df_equipos, fecha, k=20): 
    cols = ['pts_home', 'pts_away', 'wl_home', 'team_id_home', 'team_id_away']
    df = historial_5temporadas(df_games, fecha).dropna(subset=cols) # Asegurarse de que no haya NaN en las columnas necesarias
    elos = dict.fromkeys(df_equipos['id'], 1500.0) # Elo inicial 1500
    
    # Reseteo de Elo al inicio de cada temporada(FiveThirtyEight)
    def reset_elo_temporada(elos):
        for equipo_id in elos.keys():
            elos[equipo_id] = elos[equipo_id] * 0.75 + 1500 * 0.25 
        return elos

    # Valor de season_id en la primera fila
    temporada_actual = df.iloc[0]['season_id']

    for i, fila in df.iterrows():
        id_h, id_a = fila['team_id_home'], fila['team_id_away']
        pts_h, pts_a = fila['pts_home'], fila['pts_away']
        
        # Reseteo de Elo al inicio de cada temporada
        if fila['season_id'] != temporada_actual:
            elos = reset_elo_temporada(elos)
            temporada_actual = fila['season_id']

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
        
    return elos

def calcular_elos_fg3(df_games, df_equipos, fecha, k=20):
    cols = ['fg3_pct_home', 'fg3_pct_away', 'fg3m_home', 'fg3m_away', 'team_id_home', 'team_id_away']
    df = historial_5temporadas(df_games, fecha).dropna(subset=cols)
    # Inicialización de ambos ELOs a 1500
    elo_ofensivo = dict.fromkeys(df_equipos['id'], 1500.0)
    elo_defensivo = dict.fromkeys(df_equipos['id'], 1500.0)

    for i, fila in df.iterrows():
        id_h, id_a = fila['team_id_home'], fila['team_id_away']
        
        # ELO OFENSIVO, EFICACIA: Quien tiene mejor porcentaje de triples anotados
        if fila['fg3_pct_home'] > fila['fg3_pct_away']:
            win_off_h = 1 
        elif fila['fg3_pct_home'] < fila['fg3_pct_away']:
            win_off_h = 0 
        else:
            # Empate en porcentaje, decidimos por cantidad de triples metidos
            win_off_h = 1 if fila['fg3m_home'] > fila['fg3m_away'] else 0
        
        # Si empatan en todo, se considera empate ofensivo
        if fila['fg3_pct_home'] == fila['fg3_pct_away'] and fila['fg3m_home'] == fila['fg3m_away']:
            win_off_h = 0.5

        exp_off_h = prob_esperada(elo_ofensivo[id_h], elo_ofensivo[id_a])
        ajuste_off = k * (win_off_h - exp_off_h)
        elo_ofensivo[id_h] += ajuste_off
        elo_ofensivo[id_a] -= ajuste_off

        # ELO DEFENSIVO, RESISTENCIA: Quien permite menos triples anotados
        if fila['fg3m_away'] < fila['fg3m_home']:
            win_def_h = 1  
        elif fila['fg3m_away'] > fila['fg3m_home']:
            win_def_h = 0  
        else:
            # Empate en triples metidos, decidimos por porcentaje permitido
            win_def_h = 1 if fila['fg3_pct_away'] < fila['fg3_pct_home'] else 0
            
        # Si empatan en todo, se considera empate defensivo
        if fila['fg3m_away'] == fila['fg3m_home'] and fila['fg3_pct_away'] == fila['fg3_pct_home']:
            win_def_h = 0.5
        exp_def_h = prob_esperada(elo_defensivo[id_h], elo_defensivo[id_a])
        ajuste_def = k * (win_def_h - exp_def_h)
        elo_defensivo[id_h] += ajuste_def
        elo_defensivo[id_a] -= ajuste_def
        
    return elo_ofensivo, elo_defensivo

def calcular_elos_fg(df_games, df_equipos, fecha, k=20):
    cols = ['fg_pct_home', 'fg_pct_away', 'fgm_home', 'fgm_away', 'team_id_home', 'team_id_away']
    df = historial_5temporadas(df_games, fecha).dropna(subset=cols)
    # Inicialización de ambos ELOs a 1500
    elo_ofensivo = dict.fromkeys(df_equipos['id'], 1500.0)
    elo_defensivo = dict.fromkeys(df_equipos['id'], 1500.0)

    for i, fila in df.iterrows():
        id_h, id_a = fila['team_id_home'], fila['team_id_away']
        
        # ELO OFENSIVO, EFICACIA: Quien tiene mejor porcentaje de tiros de campo anotados
        if fila['fg_pct_home'] > fila['fg_pct_away']:
            win_off_h = 1  
        elif fila['fg_pct_home'] < fila['fg_pct_away']:
            win_off_h = 0  
        else:
            win_off_h = 1 if fila['fgm_home'] > fila['fgm_away'] else 0
        
        # Si empatan en todo, se considera empate ofensivo
        if fila['fg_pct_home'] == fila['fg_pct_away'] and fila['fgm_home'] == fila['fgm_away']:
            win_off_h = 0.5

        exp_off_h = prob_esperada(elo_ofensivo[id_h], elo_ofensivo[id_a])
        ajuste_off = k * (win_off_h - exp_off_h)
        elo_ofensivo[id_h] += ajuste_off
        elo_ofensivo[id_a] -= ajuste_off

        # ELO DEFENSIVO, RESISTENCIA: Quien permite menos tiros de campo anotados
        if fila['fgm_away'] < fila['fgm_home']:
            win_def_h = 1 
        elif fila['fgm_away'] > fila['fgm_home']:
            win_def_h = 0 
        else:
            # Empate en triples metidos, decidimos por porcentaje permitido
            win_def_h = 1 if fila['fg_pct_away'] < fila['fg_pct_home'] else 0  
        # Si empatan en todo, se considera empate defensivo
        if fila['fgm_away'] == fila['fgm_home'] and fila['fg_pct_away'] == fila['fg_pct_home']:
            win_def_h = 0.5

        exp_def_h = prob_esperada(elo_defensivo[id_h], elo_defensivo[id_a])
        ajuste_def = k * (win_def_h - exp_def_h)
        elo_defensivo[id_h] += ajuste_def
        elo_defensivo[id_a] -= ajuste_def
        
    return elo_ofensivo, elo_defensivo

df_partidos = pd.read_csv('csv/game.csv')
df_equipos = pd.read_csv('csv/team.csv')

# Descartar partidos que no sean Regular Season o Playoffs
df_partidos = df_partidos[df_partidos['season_type'].isin(['Regular Season', 'Playoffs'])]

# Obtener los ELOs justo antes de la fecha indicada
fecha = '2018-12-25'
elos1 = calcular_elo_nba(df_partidos, df_equipos, fecha)
elos2 = calcular_elo_nba_2(df_partidos, df_equipos, fecha)
elos_off_fg3, elos_def_fg3 = calcular_elos_fg3(df_partidos, df_equipos, fecha)
elos_off_fg, elos_def_fg = calcular_elos_fg(df_partidos, df_equipos, fecha)

# Ranking de equipos según los distintos ELOs
df_1 = pd.DataFrame(list(elos1.items()), columns=['id', 'elo'])
df_1 = df_1.merge(df_equipos[['id', 'full_name']], on='id').sort_values('elo', ascending=False)

df_2 = pd.DataFrame(list(elos2.items()), columns=['id', 'elo'])
df_2 = df_2.merge(df_equipos[['id', 'full_name']], on='id').sort_values('elo', ascending=False)

df_off_fg3 = pd.DataFrame(list(elos_off_fg3.items()), columns=['id', 'elo_off_fg3'])
df_off_fg3 = df_off_fg3.merge(df_equipos[['id', 'full_name']], on='id').sort_values('elo_off_fg3', ascending=False)
df_def_fg3 = pd.DataFrame(list(elos_def_fg3.items()), columns=['id', 'elo_def_fg3'])
df_def_fg3 = df_def_fg3.merge(df_equipos[['id', 'full_name']], on='id').sort_values('elo_def_fg3', ascending=False)

df_off_fg = pd.DataFrame(list(elos_off_fg.items()), columns=['id', 'elo_off_fg'])
df_off_fg = df_off_fg.merge(df_equipos[['id', 'full_name']], on='id').sort_values('elo_off_fg', ascending=False)
df_def_fg = pd.DataFrame(list(elos_def_fg.items()), columns=['id', 'elo_def_fg'])
df_def_fg = df_def_fg.merge(df_equipos[['id', 'full_name']], on='id').sort_values('elo_def_fg', ascending=False)

#Rankings elo1
print(f"Ranking equipos según elo1 el {fecha}:")
print(df_1[['full_name', 'elo']].to_string(index=False))

#Rankings elo2
print(f"\nRanking equipos según elo2 el {fecha}:")
print(df_2[['full_name', 'elo']].to_string(index=False))

#Rankings FG3
print(f"\nRanking equipos según elo ofensivo FG3 el {fecha}:")
print(df_off_fg3[['full_name', 'elo_off_fg3']].to_string(index=False))
print(f"\nRanking equipos según elo defensivo FG3 el {fecha}:")
print(df_def_fg3[['full_name', 'elo_def_fg3']].to_string(index=False))

#Rankings FG
print(f"\nRanking equipos según elo ofensivo FG el {fecha}:")
print(df_off_fg[['full_name', 'elo_off_fg']].to_string(index=False))
print(f"\nRanking equipos según elo defensivo FG el {fecha}:")
print(df_def_fg[['full_name', 'elo_def_fg']].to_string(index=False))

