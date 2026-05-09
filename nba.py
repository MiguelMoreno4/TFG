import os
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

SEED = 42
MIN_HISTORY_GAMES = 5
ROLLING_WINDOWS = (5, 10)
ELO_K = 20
HOME_ADVANTAGE_ELO = 65
SEASON_ELO_REGRESSION = 0.25


def load_csvs(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(base_dir / "game.csv")
    line_score = pd.read_csv(base_dir / "line_score.csv")
    other_stats = pd.read_csv(base_dir / "other_stats.csv")

    for dataframe in (games, line_score, other_stats):
        dataframe.columns = dataframe.columns.str.lower()

    games["game_date"] = pd.to_datetime(games["game_date"])
    if "game_date_est" in line_score.columns:
        line_score["game_date_est"] = pd.to_datetime(line_score["game_date_est"])

    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return games, line_score, other_stats


def build_team_game_table(
    games: pd.DataFrame,
    line_score: pd.DataFrame,
    other_stats: pd.DataFrame,
) -> pd.DataFrame:
    games = games.copy()
    games["home_win"] = (games["pts_home"] > games["pts_away"]).astype(int)

    base_stats = ["fg_pct", "fg3_pct", "ft_pct", "reb", "ast", "stl", "blk", "tov", "pf", "pts"]
    team_frames = []

    for side, opponent_side in (("home", "away"), ("away", "home")):
        team_frame = pd.DataFrame(
            {
                "game_id": games["game_id"],
                "game_date": games["game_date"],
                "season_id": games["season_id"],
                "season_type": games["season_type"],
                "team_id": games[f"team_id_{side}"],
                "opponent_id": games[f"team_id_{opponent_side}"],
                "is_home": 1 if side == "home" else 0,
                "win": games["home_win"] if side == "home" else 1 - games["home_win"],
                "points_for": games[f"pts_{side}"],
                "points_against": games[f"pts_{opponent_side}"],
            }
        )

        for stat_name in base_stats:
            team_frame[stat_name] = games[f"{stat_name}_{side}"]
            team_frame[f"opp_{stat_name}"] = games[f"{stat_name}_{opponent_side}"]

        team_frames.append(team_frame)

    team_games = pd.concat(team_frames, ignore_index=True)

    quarter_stats = [
        "pts_qtr1",
        "pts_qtr2",
        "pts_qtr3",
        "pts_qtr4",
        "pts_ot1",
        "pts_ot2",
        "pts_ot3",
        "pts_ot4",
        "pts_ot5",
        "pts_ot6",
        "pts_ot7",
        "pts_ot8",
        "pts_ot9",
        "pts_ot10",
    ]
    quarter_frames = []
    for side in ("home", "away"):
        quarter_frame = line_score[["game_id", f"team_id_{side}"]].copy()
        quarter_frame = quarter_frame.rename(columns={f"team_id_{side}": "team_id"})
        for stat_name in quarter_stats:
            source_col = f"{stat_name}_{side}"
            if source_col in line_score.columns:
                quarter_frame[stat_name] = line_score[source_col]
        quarter_frames.append(quarter_frame)
    quarter_stats_by_team = pd.concat(quarter_frames, ignore_index=True)

    extra_stats = [
        "pts_paint",
        "pts_2nd_chance",
        "pts_fb",
        "largest_lead",
        "lead_changes",
        "times_tied",
        "team_turnovers",
        "total_turnovers",
        "team_rebounds",
        "pts_off_to",
    ]
    other_frames = []
    for side in ("home", "away"):
        other_frame = other_stats[["game_id", f"team_id_{side}"]].copy()
        other_frame = other_frame.rename(columns={f"team_id_{side}": "team_id"})
        for stat_name in extra_stats:
            source_col = f"{stat_name}_{side}"
            if source_col in other_stats.columns:
                other_frame[stat_name] = other_stats[source_col]
        other_frames.append(other_frame)
    other_stats_by_team = pd.concat(other_frames, ignore_index=True)

    team_games = team_games.merge(quarter_stats_by_team, on=["game_id", "team_id"], how="left")
    team_games = team_games.merge(other_stats_by_team, on=["game_id", "team_id"], how="left")
    team_games = team_games.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    return team_games


def add_historical_features(team_games: pd.DataFrame) -> pd.DataFrame:
    team_games = team_games.copy()
    protected_columns = {
        "game_id",
        "game_date",
        "season_id",
        "season_type",
        "team_id",
        "opponent_id",
        "is_home",
        "win",
    }
    stat_columns = [
        column
        for column in team_games.columns
        if column not in protected_columns and pd.api.types.is_numeric_dtype(team_games[column])
    ]

    grouped_by_team = team_games.groupby("team_id", sort=False)
    grouped_by_team_season = team_games.groupby(["team_id", "season_id"], sort=False)
    grouped_by_team_venue = team_games.groupby(["team_id", "is_home"], sort=False)

    for column in stat_columns:
        for window in ROLLING_WINDOWS:
            feature_name = f"{column}_avg_{window}"
            team_games[feature_name] = grouped_by_team[column].transform(
                lambda series, w=window: series.shift(1).rolling(w, min_periods=1).mean()
            )

    team_games["win_pct_10"] = grouped_by_team["win"].transform(
        lambda series: series.shift(1).rolling(10, min_periods=1).mean()
    )
    team_games["venue_win_pct_10"] = grouped_by_team_venue["win"].transform(
        lambda series: series.shift(1).rolling(10, min_periods=1).mean()
    )
    team_games["games_played"] = grouped_by_team.cumcount()
    team_games["season_games_played"] = grouped_by_team_season.cumcount()
    team_games["days_since_last"] = grouped_by_team["game_date"].diff().dt.days
    team_games["days_since_last"] = team_games["days_since_last"].fillna(7).clip(0, 14)
    return team_games


def add_elo_features(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    games["home_win"] = (games["pts_home"] > games["pts_away"]).astype(int)
    ratings: dict[int, float] = {}
    current_season = None
    home_elos = []
    away_elos = []

    for row in games.itertuples(index=False):
        if current_season is None:
            current_season = row.season_id
        elif row.season_id != current_season:
            ratings = {
                team_id: rating * (1.0 - SEASON_ELO_REGRESSION) + 1500.0 * SEASON_ELO_REGRESSION
                for team_id, rating in ratings.items()
            }
            current_season = row.season_id

        home_rating = ratings.get(row.team_id_home, 1500.0)
        away_rating = ratings.get(row.team_id_away, 1500.0)
        home_elos.append(home_rating)
        away_elos.append(away_rating)

        expected_home = 1.0 / (1.0 + 10 ** (-(home_rating + HOME_ADVANTAGE_ELO - away_rating) / 400.0))
        actual_home = float(row.home_win)
        margin = abs(float(row.pts_home) - float(row.pts_away))
        multiplier = np.log(max(margin, 1) + 1.0) * (2.2 / (((home_rating - away_rating) * 0.001) + 2.2))
        rating_change = ELO_K * multiplier * (actual_home - expected_home)

        ratings[row.team_id_home] = home_rating + rating_change
        ratings[row.team_id_away] = away_rating - rating_change

    games["home_elo"] = home_elos
    games["away_elo"] = away_elos
    games["elo_diff"] = games["home_elo"] - games["away_elo"]
    return games


def build_model_table(games: pd.DataFrame, team_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    games = add_elo_features(games)

    derived_columns = [
        column
        for column in team_games.columns
        if column.endswith("_avg_5")
        or column.endswith("_avg_10")
        or column in {"win_pct_10", "venue_win_pct_10", "games_played", "season_games_played", "days_since_last"}
    ]

    pregame_team_games = team_games[team_games["games_played"] >= MIN_HISTORY_GAMES].copy()
    home_features = pregame_team_games[pregame_team_games["is_home"] == 1][["game_id", *derived_columns]].add_prefix("home_")
    away_features = pregame_team_games[pregame_team_games["is_home"] == 0][["game_id", *derived_columns]].add_prefix("away_")

    model_table = games[
        ["game_id", "game_date", "season_type", "home_elo", "away_elo", "elo_diff", "home_win"]
    ].copy()
    model_table["is_playoffs"] = (model_table["season_type"] != "Regular Season").astype(int)
    model_table = model_table.merge(home_features, left_on="game_id", right_on="home_game_id", how="inner")
    model_table = model_table.merge(away_features, left_on="game_id", right_on="away_game_id", how="inner")

    model_table["rest_diff"] = model_table["home_days_since_last"] - model_table["away_days_since_last"]

    diff_columns = []
    for column in derived_columns:
        home_column = f"home_{column}"
        away_column = f"away_{column}"
        if home_column in model_table.columns and away_column in model_table.columns:
            diff_name = f"diff_{column}"
            model_table[diff_name] = model_table[home_column] - model_table[away_column]
            diff_columns.append(diff_name)

    selected_features = [
        "home_elo",
        "away_elo",
        "elo_diff",
        "is_playoffs",
        "rest_diff",
    ]
    selected_features.extend([f"home_{column}" for column in derived_columns])
    selected_features.extend([f"away_{column}" for column in derived_columns])
    selected_features.extend(diff_columns)

    X = model_table[selected_features].copy()
    y = model_table["home_win"].copy()
    chronological_order = np.argsort(model_table["game_date"].to_numpy())
    X = X.iloc[chronological_order].reset_index(drop=True)
    y = y.iloc[chronological_order].reset_index(drop=True)
    return X, y


def split_time_based(X: pd.DataFrame, y: pd.Series):
    train_end = int(len(X) * 0.7)
    validation_end = int(len(X) * 0.85)

    X_train = X.iloc[:train_end].copy()
    X_validation = X.iloc[train_end:validation_end].copy()
    X_test = X.iloc[validation_end:].copy()
    y_train = y.iloc[:train_end].copy()
    y_validation = y.iloc[train_end:validation_end].copy()
    y_test = y.iloc[validation_end:].copy()
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def transform_features(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)
    X_validation_imputed = imputer.transform(X_validation)
    X_test_imputed = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_validation_scaled = scaler.transform(X_validation_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    return X_train_scaled, X_validation_scaled, X_test_scaled, imputer, scaler


def build_candidate_models() -> dict[str, object]:
    return {
        "logreg": LogisticRegression(max_iter=1500, C=0.5, random_state=SEED),
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            min_samples_leaf=50,
            random_state=SEED,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=SEED,
        ),
    }


def predict_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    raw_scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-raw_scores))


def select_best_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_validation: np.ndarray,
    y_validation: pd.Series,
):
    best_model_name = None
    best_model = None
    best_metrics = None

    print("Comparando modelos candidatos...")
    for model_name, candidate_model in build_candidate_models().items():
        fitted_model = clone(candidate_model)
        fitted_model.fit(X_train, y_train)
        validation_probabilities = predict_scores(fitted_model, X_validation)
        validation_predictions = (validation_probabilities >= 0.5).astype(int)
        validation_accuracy = accuracy_score(y_validation, validation_predictions)
        validation_auc = roc_auc_score(y_validation, validation_probabilities)
        metrics = {
            "accuracy": validation_accuracy,
            "auc": validation_auc,
        }
        print(
            f"{model_name}: val_accuracy={validation_accuracy:.4f} val_auc={validation_auc:.4f}"
        )

        if best_metrics is None or validation_auc > best_metrics["auc"]:
            best_model_name = model_name
            best_model = fitted_model
            best_metrics = metrics

    return best_model_name, best_model, best_metrics


def fit_final_preprocessors(X_train_validation: pd.DataFrame, X_test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_validation_imputed = imputer.fit_transform(X_train_validation)
    X_test_imputed = imputer.transform(X_test)
    X_train_validation_scaled = scaler.fit_transform(X_train_validation_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    return X_train_validation_scaled, X_test_scaled, imputer, scaler


def plot_feature_importance(
    model,
    X_test: np.ndarray,
    y_test: pd.Series,
    feature_names: list[str],
    output_path: Path,
) -> None:
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = np.abs(model.coef_[0])
    else:
        sample_size = min(len(X_test), 3000)
        importance = permutation_importance(
            model,
            X_test[:sample_size],
            y_test.iloc[:sample_size],
            n_repeats=5,
            random_state=SEED,
            n_jobs=-1,
        )
        importance_values = importance.importances_mean

    importance_frame = (
        pd.DataFrame({"feature": feature_names, "importance": importance_values})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    fig, axis = plt.subplots(figsize=(11, 7))
    axis.barh(importance_frame["feature"][::-1], importance_frame["importance"][::-1], color="#1f77b4")
    axis.set_title("Top 15 feature importances")
    axis.set_xlabel("Importance")
    axis.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    try:
        games, line_score, other_stats = load_csvs(base_dir)
    except FileNotFoundError as error:
        print(f"Error al cargar los CSV: {error}")
        raise SystemExit(1) from error

    print("Construyendo variables historicas...")
    team_games = build_team_game_table(games, line_score, other_stats)
    team_games = add_historical_features(team_games)
    X, y = build_model_table(games, team_games)

    if len(X) == 0:
        print("No hay suficientes partidos con historial previo para entrenar el modelo.")
        raise SystemExit(1)

    X_train, X_validation, X_test, y_train, y_validation, y_test = split_time_based(X, y)
    X_train_scaled, X_validation_scaled, X_test_scaled, _, _ = transform_features(
        X_train,
        X_validation,
        X_test,
    )

    baseline_accuracy = max(y_test.mean(), 1.0 - y_test.mean())
    print(
        f"Entrenando con {len(X_train)} partidos, validando con {len(X_validation)} y evaluando con {len(X_test)}..."
    )
    best_model_name, _, best_validation_metrics = select_best_model(
        X_train_scaled,
        y_train,
        X_validation_scaled,
        y_validation,
    )

    X_train_validation = pd.concat([X_train, X_validation], axis=0)
    y_train_validation = pd.concat([y_train, y_validation], axis=0)
    X_train_validation_scaled, X_test_scaled, imputer, scaler = fit_final_preprocessors(
        X_train_validation,
        X_test,
    )
    final_model = clone(build_candidate_models()[best_model_name])
    final_model.fit(X_train_validation_scaled, y_train_validation)

    test_probabilities = predict_scores(final_model, X_test_scaled)
    test_predictions = (test_probabilities >= 0.5).astype(int)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_auc = roc_auc_score(y_test, test_probabilities)
    print(f"Mejor modelo por validacion: {best_model_name}")
    print(
        f"Validacion del modelo elegido: accuracy={best_validation_metrics['accuracy']:.4f} auc={best_validation_metrics['auc']:.4f}"
    )
    print(f"Baseline mayoritaria: {baseline_accuracy:.4f}")
    print(f"Accuracy test: {test_accuracy:.4f}")
    print(f"AUC test: {test_auc:.4f}")

    model_bundle = {
        "model_name": best_model_name,
        "model": final_model,
        "imputer": imputer,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "test_accuracy": float(test_accuracy),
        "test_auc": float(test_auc),
    }
    joblib.dump(model_bundle, base_dir / "nba_winner_model.joblib")
    joblib.dump(imputer, base_dir / "imputer.save")
    joblib.dump(scaler, base_dir / "scaler.save")
    joblib.dump(list(X.columns), base_dir / "feature_columns.save")
    plot_feature_importance(final_model, X_test_scaled, y_test, list(X.columns), base_dir / "experimento7.png")
    print("Modelo y artefactos guardados correctamente.")


if __name__ == "__main__":
    main()