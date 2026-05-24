Experimento D2

Este experimento representa una evolución directa del Experimento D1. Mientras que el enfoque inicial se centraba en una arquitectura de red neuronal sencilla con variables limitadas, el D2 amplía significativamente el análisis mediante la implementación de modelos de clasificación tipo ensamble y un sistema de calibración basado en rating Elo.

## 1. Descripción del Experimento
El experimento D2 propone un enfoque híbrido que combina la potencia de los algoritmos de Machine Learning tabular con la solidez estadística del sistema Elo. A diferencia de los intentos previos, este modelo no intenta predecir la puntuación exacta sino que se formula como un problema de clasificación binaria para determinar el ganador local.
### Mejoras clave respecto a D1
Ingeniería de Características Avanzada: Se introducen promedios móviles rolling windows de 5 y 10 partidos para capturar rachas de rendimiento recientes.
Integración de Elo: Se utiliza el rating Elo como una característica dinámica que mide la fuerza relativa de los equipos justo antes de cada encuentro.
Modelos de Ensamble: Se evalúan y comparan tres algoritmos robustos: Regresión Logística, Random Forest y HistGradientBoosting.
## 2. Metodología y Preparación de Datos
El flujo de trabajo implementado está diseñado rigurosamente para evitar la fuga de datos y asegurar la validez temporal del modelo:
Variables Históricas: El modelo calcula estadísticas prepartido como los puntos a favor, en contra, rebotes, asistencias, etc. Basadas únicamente en el historial previo al día del encuentro.
Diferenciales de Equipo: Se calculan las diferencias entre las métricas del equipo local y visitante como diff_win_pct_10, permitiendo al modelo entender la ventaja relativa.
División Temporal: Se emplea un esquema de validación temporal 70% entrenamiento, 15% validación, 15% test respetando el orden cronológico de las temporadas del 1946 al 2023.
Transformación: Se aplica un escalado estándar StandardScaler y una imputación de valores nulos mediante la mediana para garantizar la estabilidad de los algoritmos.
## 3. Resultados y Comparativa

El experimento demuestra que los modelos basados en árboles y la inclusión de métricas de rating ofrecen un rendimiento superior a las redes neuronales simples del experimento D1.

| Modelo | Accuracy Validación | AUC Validación |
| :--- | :---: | :---: |
| **Logistic Regression** | 0.6480 | 0.7120 |
| **HistGradientBoosting** | 0.6545 | 0.7215 |
| **Random Forest** | **0.6610** | **0.7285** |

El modelo Random Forest destacó como el más equilibrado logrando precisiones superiores al 65% y batiendo consistentemente la línea base de la localía.
## 4. Análisis de Importancia de Variables
Una de las grandes aportaciones de D2 es la interpretabilidad. El análisis de importancia de características (feature importance) reveló que las variables más determinantes para predecir al ganador son:

1. **Elo Diff:** La diferencia de rating Elo entre ambos equipos.
2. **Win Pct 10:** El porcentaje de victorias en los últimos 10 partidos (racha reciente).
3. **Puntos en la pintura (promedio):** Métrica que refleja el dominio físico reciente en la zona.

## Conclusión

El experimento D2 confirma que en la predicción de la NBA, la calidad y el contexto de las variables (como el rating Elo) tienen un impacto más significativo que la complejidad arquitectónica del modelo. Este enfoque no solo logra superar la barrera del azar, sino que proporciona una base sólida para sistemas de predicción deportiva profesional.
