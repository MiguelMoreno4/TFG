**Versión de Python utilizada:** `3.12.10`

## Requisitos

1. **Conjunto de datos (Dataset):** Descarga los conjuntos de datos en formato CSV que se encuentran en: [Kaggle - Basketball Dataset](https://www.kaggle.com/datasets/wyattowalsh/basketball)
2. **Entorno:** Crea un entorno virtual de Python e instala las dependencias necesarias. Puedes instalarlas usando:
   ```bash
   pip install pandas numpy tensorflow scikit-learn
   ```

## Contenido

### Experimento C1

* `elo2.py` -> Cálculo de las variantes de Elo y preprocesamiento de datos.
    * **Constantes:**
        * `FECHA`: Contiene la fecha del último partido de la temporada 2018-2019. Los cálculos de Elo se realizarán para todos los partidos de la Temporada Regular y de los PlayOffs de las 5 temporadas anteriores a esta fecha.
        * `SOURCE_FOLDER`: Contiene la ruta a la carpeta que contiene los CSVs del conjunto de datos de Kaggle.
        * `FOLDER`: Contiene la ruta de destino para el archivo CSV generado con los datos preprocesados.

* `v1.py` -> Ejecución del modelo de red neuronal. Muestra los resultados y guarda cada predicción para cada modelo en formato CSV dentro de la carpeta `/resultados`.

### Extra

* `elo.py` -> Versión beta de las variantes de Elo 1 y 2, cálculos de defensa/ataque y preprocesamiento de datos. Muestra el ranking de Elo ordenado para cada tipo de Elo.

* `v1_1.py` -> Clon de `v1.py`. Sustituye la red neuronal original por un simple predictor de ganadores. Muestra los resultados y guarda cada predicción para cada modelo en formato csv dentro de la carpeta `/resultados2`. El % de Precisión del Ganador es muy similar al de la red neuronal original de `v1.py`, por lo que se excluyó del experimento principal.


# English

**Python version used:** `3.12.10`

## Requirements

1. **Dataset:** Download the CSV datasets found at: [Kaggle - Basketball Dataset](https://www.kaggle.com/datasets/wyattowalsh/basketball)
2. **Environment:** Create a Python virtual environment and install the required dependencies. You can install them using:
   ```bash
   pip install pandas numpy tensorflow scikit-learn
   ```

## Contents

### Experiment C1

* `elo2.py` -> Elo variants calculation and data preprocessing.
    * **Constants:**
        * `FECHA`: Contains the date of the last 2018-2019 season match. Elo calculations will be done for all Regular Season and PlayOff matches 5 seasons prior to this date.
        * `SOURCE_FOLDER`: Contains the route to the folder containing the CSVs from the Kaggle dataset.
        * `FOLDER`: Contains the destination route for the generated CSV with the preprocessed data.

* `v1.py` -> Neural network model execution. Shows results and saves each prediction for each model in CSV format inside the `/resultados` folder.

### Extra

* `elo.py` -> Beta version of Elo variants 1 & 2, defense/offense calculations, and data preprocessing. Shows the ordered Elo ranking for each Elo type.

* `v1_1.py` -> Clone of `v1.py`. Substitutes the original neural network for a simple winner predictor.  Shows results and saves each prediction for each model in csv format inside '/resultados2' folder. The Winner Accuracy % is very similar to the original `v1.py` neural network, thus it was excluded from the main experiment.