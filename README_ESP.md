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