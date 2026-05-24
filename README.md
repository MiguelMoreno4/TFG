# Experimento D1
Este experimento representa una evolución importate sobre el Experimento D. Mientras que el modelo base (D) se centraba en estadísticas simples de rendimiento, el Experimento D1 introduce métricas de fuerza relativa y ventanas temporales para capturar la dinámica de la competición.
## 1. Comparativa de Modelos D con D1

### Experimento D (Línea Base)
*   **Enfoque**: Clasificación binaria basada en estadísticas puntuales del partido (rebotes, asistencias, porcentajes de tiro).
*   **Arquitectura**: Red Neuronal Artificial (MLP) secuencial con capas de 16 y 8 neuronas.
*   **Limitación**: El modelo presentaba una precisión cercana al **51.8%**, apenas superior al azar debido a que las variables aisladas no capturaban la consistencia de los equipos.

### Experimento D1 (Mejora)
*   **Enfoque**: Modelado híbrido que combina Rating Elo con promedios móviles.
*   **Variables Clave**:
    *   **Diferencial de Elo**: Captura la calidad relativa de los equipos antes del salto inicial.
    *   **Promedios Móviles (5 y 10 partidos)**: Suaviza la variabilidad de las estadísticas básicas (puntos, rebotes, asistencias) para reflejar el estado de forma actual.
    *   **Ventaja de Localía y Días de Descanso**: Incorpora el factor cansancio y el impacto del pabellón.
## 2. Arquitectura del Sistema D1
El código implementa un pipeline completo de Machine Learning distribuido en las siguientes fases:
1.  **Cálculo de Elo Dinámico**: Implementación de la fórmula de FiveThirtyEight con ajustes por margen de victoria (MOV) y regresión de temporada (25%).
2.  **Ingeniería de Características**: 
    *   Generación de medias móviles para 45+ variables estadísticas.
    *   Cálculo de diferenciales (Home - Away) para normalizar la comparación.
3.  **Selección de Modelos**: Comparación automatizada entre:
    *   **Logistic Regression**: Base lineal.
    *   **Random Forest**: Captura de relaciones no lineales.
    *   **HistGradientBoosting**: Optimizado para grandes volúmenes de datos.
## 3. Resultados
Los resultados obtenidos en el Experimento D1 muestran un salto grande respecto a la versión anterior:

| Métrica | Experimento D | Experimento D1 (Mejor Modelo) |
| :--- | :---: | :---: |
| **Accuracy (Test)** | ~52% | **>65%** |
| **Variables** | 5 (Estadísticas simples) | 100+ (Elo + Historial) |
| **Metodología** | Red Neuronal Simple | Ensamble / Boosting |

### 4. Análisis de Importancia
El análisis mediante Feature Importance revela que el Elo Diff (Diferencia de Rating Elo) y el Win Pct 10 (Porcentaje de victorias recientes) son los predictores más potentes, validando la hipótesis de que el contexto histórico supera a la estadística puntual del partido.
