# Análisis Exploratorio de Correlaciones para Predicción NBA



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# PARÁMETROS GENERALES


CSV_GAME = "../../../../data/inputs/csv/game.csv"
CSV_LINE = "../../../../data/inputs/csv/line_score.csv"
CSV_OTHER = "../../../../data/inputs/csv/other_stats.csv"
CSV_COMMON_PLAYER_INFO = "../../../../data/inputs/csv/common_player_info.csv"
CSV_DRAFT_COMBINE = "../../../../data/inputs/csv/draft_combine_stats.csv"
CSV_DRAFT_HISTORY = "../../../../data/inputs/csv/draft_history.csv"
CSV_GAME_INFO = "../../../../data/inputs/csv/game_info.csv"
CSV_GAME_SUMMARY = "../../../../data/inputs/csv/game_summary.csv"
CSV_INACTIVE_PLAYERS = "../../../../data/inputs/csv/inactive_players.csv"
CSV_OFFICIALS = "../../../../data/inputs/csv/officials.csv"
CSV_PLAY_BY_PLAY = "../../../../data/inputs/csv/play_by_play.csv"
CSV_PLAYER = "../../../../data/inputs/csv/player.csv"
CSV_TEAM = "../../../../data/inputs/csv/team.csv"


# CARGA DE DATOS
games = pd.read_csv(CSV_GAME)
line_score = pd.read_csv(CSV_LINE)
other_stats = pd.read_csv(CSV_OTHER)

common_player_info = pd.read_csv(CSV_COMMON_PLAYER_INFO)
draft_combine_stats = pd.read_csv(CSV_DRAFT_COMBINE)
draft_history = pd.read_csv(CSV_DRAFT_HISTORY)
game_info = pd.read_csv(CSV_GAME_INFO)
game_summary = pd.read_csv(CSV_GAME_SUMMARY)
inactive_players = pd.read_csv(CSV_INACTIVE_PLAYERS)
officials = pd.read_csv(CSV_OFFICIALS)
play_by_play = pd.read_csv(CSV_PLAY_BY_PLAY)
player = pd.read_csv(CSV_PLAYER)
team = pd.read_csv(CSV_TEAM)



# PREPROCESADO BASE
games['game_date'] = pd.to_datetime(games['game_date'], errors='coerce')
games = games.sort_values('game_date').reset_index(drop=True)

games['home_win'] = (games['pts_home'] > games['pts_away']).astype(int)

print("Dataset games cargado correctamente")
print(games.shape)



# FUNCION AUXILIAR DE CORRELACIoN

def analizar_correlaciones(df, nombre_tabla):

    print(f"\nAnalizando tabla: {nombre_tabla}")

    df_ = df.copy()

    # Pasar columnas a minúsculas
    df_.columns = [c.lower() for c in df_.columns]

    # Verificar existencia de game_id
    if 'game_id' not in df_.columns:
        print(f"{nombre_tabla}: no contiene game_id")
        return

    # Merge con home_win
    analysis = df_.merge(
        games[['game_id', 'home_win']],
        on='game_id',
        how='left'
    )

    # Selección numérica
    numeric_df = analysis.select_dtypes(include=[np.number])

    # Comprobar home_win
    if 'home_win' not in numeric_df.columns:
        print(f"{nombre_tabla}: home_win no encontrado")
        return

    # Correlación
    corr = numeric_df.corr()

    if 'home_win' not in corr.columns:
        print(f"{nombre_tabla}: no se pudo calcular correlación")
        return

    corr_target = corr['home_win'].drop('home_win') \
                                          .sort_values(ascending=False)

    if len(corr_target) == 0:
        print(f"{nombre_tabla}: sin variables numéricas útiles")
        return

   #GRÄFICO
    plt.figure(figsize=(8, max(6, len(corr_target)*0.35)))

    sns.barplot(
        x=corr_target.values,
        y=corr_target.index
    )

    plt.title(f"Correlación ({nombre_tabla}) con victoria local")
    plt.xlabel("Coeficiente de correlación")
    plt.tight_layout()
    plt.show()



    # TOP 10 CORRELACIONES ABSOLUTAS

    top10 = corr_target.abs().sort_values(ascending=False).head(10)

    plt.figure(figsize=(8,6))

    sns.barplot(
        x=top10.values,
        y=top10.index
    )

    plt.title(f"Top 10 variables más correlacionadas (|corr|) - {nombre_tabla}")
    plt.xlabel("Correlación absoluta")
    plt.tight_layout()
    plt.show()

    print(f"\nTop 10 correlaciones absolutas - {nombre_tabla}")
    print(top10)


# ==============================================================
# 1. CORRELACIONES - GAMES
# ==============================================================

print("\n==================== GAMES ====================")

games_num = games.select_dtypes(include=[np.number])

corr_games = games_num.corr()

corr_target_games = corr_games['home_win'].drop('home_win') \
                                         .sort_values(ascending=False)

# ----- Gráfico completo -----
plt.figure(figsize=(8,10))

sns.barplot(
    x=corr_target_games.values,
    y=corr_target_games.index
)

plt.title("Correlación (games) con victoria local")
plt.xlabel("Coeficiente de correlación")
plt.tight_layout()
plt.show()


# ----- Top 10 -----

top10_games = corr_target_games.abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(8,6))

sns.barplot(
    x=top10_games.values,
    y=top10_games.index
)

plt.title("Top 10 variables más correlacionadas (|corr|) - games")
plt.xlabel("Correlación absoluta")
plt.tight_layout()
plt.show()

print("Top 10 correlaciones absolutas - games")
print(top10_games)


# ==============================================================
# 2. CORRELACIONES - LINE_SCORE
# ==============================================================

print("\n==================== LINE_SCORE ====================")

line_score_analysis = line_score.merge(
    games[['game_id', 'home_win']],
    on='game_id',
    how='left'
)

line_score_num = line_score_analysis.select_dtypes(include=[np.number])

corr_line = line_score_num.corr()

corr_target_line = corr_line['home_win'].drop('home_win') \
                                        .sort_values(ascending=False)

# ----- Gráfico completo -----
plt.figure(figsize=(8,12))

sns.barplot(
    x=corr_target_line.values,
    y=corr_target_line.index
)

plt.title("Correlación (line_score) con victoria local")
plt.xlabel("Coeficiente de correlación")
plt.tight_layout()
plt.show()


# ----- Top 10 -----

top10_line = corr_target_line.abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(8,6))

sns.barplot(
    x=top10_line.values,
    y=top10_line.index
)

plt.title("Top 10 variables más correlacionadas (|corr|) - line_score")
plt.xlabel("Correlación absoluta")
plt.tight_layout()
plt.show()

print("Top 10 correlaciones absolutas - line_score")
print(top10_line)


# ==============================================================
# 3. CORRELACIONES - OTHER_STATS
# ==============================================================

print("\n==================== OTHER_STATS ====================")

other_stats_ = other_stats.copy()
other_stats_.columns = [c.lower() for c in other_stats_.columns]

other_analysis = other_stats_.merge(
    games[['game_id', 'home_win']],
    on='game_id',
    how='left'
)

other_num = other_analysis.select_dtypes(include=[np.number])

corr_other = other_num.corr()

corr_target_other = corr_other['home_win'].drop('home_win') \
                                          .sort_values(ascending=False)

# ----- Gráfico completo -----
plt.figure(figsize=(8,14))

sns.barplot(
    x=corr_target_other.values,
    y=corr_target_other.index
)

plt.title("Correlación (other_stats) con victoria local")
plt.xlabel("Coeficiente de correlación")
plt.tight_layout()
plt.show()


# ----- Top 10 -----

top10_other = corr_target_other.abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(8,6))

sns.barplot(
    x=top10_other.values,
    y=top10_other.index
)

plt.title("Top 10 variables más correlacionadas (|corr|) - other_stats")
plt.xlabel("Correlación absoluta")
plt.tight_layout()
plt.show()

print("Top 10 correlaciones absolutas - other_stats")
print(top10_other)




analizar_correlaciones(common_player_info, "common_player_info")
analizar_correlaciones(draft_combine_stats, "draft_combine_stats")
analizar_correlaciones(draft_history, "draft_history")
analizar_correlaciones(game_info, "game_info")
analizar_correlaciones(game_summary, "game_summary")
analizar_correlaciones(inactive_players, "inactive_players")
analizar_correlaciones(officials, "officials")
analizar_correlaciones(play_by_play, "play_by_play")
analizar_correlaciones(player, "player")
analizar_correlaciones(team, "team")



print("\nAnálisis completo finalizado")


