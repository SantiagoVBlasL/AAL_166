# NeuroInsight-VAE: Un Pipeline de Deep Learning para la Clasificación de la Enfermedad de Alzheimer a partir de Conectividad Funcional fMRI

## Resumen Ejecutivo

Este repositorio presenta un pipeline computacional avanzado para la clasificación de la Enfermedad de Alzheimer (EA) frente a Controles Sanos (CN), utilizando representaciones latentes extraídas de matrices de conectividad funcional dinámica (dFC) de fMRI. El núcleo de este pipeline es un **Autoencoder Variacional Convolucional (VAE)**, un modelo generativo no supervisado diseñado para aprender una representación comprimida y significativa de la compleja topología de las redes cerebrales. Estas características latentes, que capturan la esencia de las alteraciones en la conectividad, son posteriormente utilizadas para entrenar un conjunto de clasificadores supervisados, logrando una alta precisión en la distinción entre grupos. El framework está diseñado para ser robusto, reproducible y extensible, empleando validación cruzada anidada para una evaluación insesgada del rendimiento y la optimización de hiperparámetros.

---

## Marco Metodológico

El pipeline se estructura en una secuencia de etapas de procesamiento y modelado, cada una justificada por principios neurocientíficos y de aprendizaje automático.

### 1. Preprocesamiento y Construcción del Tensor de Conectividad

- **Datos de Entrada**: El pipeline opera sobre un tensor global de datos (`.npz`) que contiene las matrices de conectividad funcional para cada sujeto. Este tensor tiene una forma de `(N_sujetos, N_canales, N_ROIs, N_ROIs)`.
  - **ROIs**: Se utiliza el atlas **AAL3 con 131 ROIs**, tras un riguroso control de calidad (`check_timepoints_rois.ipynb`) que asegura la calidad de la señal y la integridad de los datos.
  - **Canales de Conectividad**: El tensor encapsula múltiples modalidades de conectividad funcional, reflejando diferentes aspectos de las interacciones cerebrales. Para la ejecución documentada, se utilizaron los canales:
    1.  `Pearson_Full_FisherZ_Signed`: Correlación de Pearson con transformación Z de Fisher.
    2.  `MI_KNN_Symmetric`: Información Mutua, una medida no lineal de dependencia.
    3.  `dFC_AbsDiffMean`: Medida de la variabilidad de la conectividad dinámica.
- **Normalización**: Se aplica una normalización `z-score` a los elementos fuera de la diagonal (`zscore_offdiag`) para cada canal de conectividad. Esta elección es crucial para estandarizar la distribución de los valores de conectividad, preservando la estructura de la diagonal (auto-conectividad) y asegurando que el VAE no sea sesgado por las diferentes escalas de las métricas de conectividad.

### 2. Autoencoder Variacional Convolucional (VAE) para Reducción de Dimensionalidad

El pilar de nuestro enfoque es el uso de un VAE convolucional para aprender una representación latente de las matrices de conectividad.

- **Justificación del VAE**:
  - **Reducción de Dimensionalidad No Lineal**: A diferencia de métodos lineales como PCA, un VAE puede capturar relaciones no lineales complejas inherentes a la conectividad cerebral.
  - **Regularización del Espacio Latente**: El VAE impone una restricción de regularización (el término de divergencia KL en la función de pérdida) que fuerza al espacio latente a seguir una distribución predefinida (Gaussiana isotrópica). Esto resulta en un espacio latente más suave y continuo, ideal para la clasificación.
  - **Capacidad Generativa**: Aunque no se explota en este pipeline para la clasificación, la capacidad del VAE para generar nuevas matrices de conectividad sintéticas abre vías para el aumento de datos y el estudio de los patrones de conectividad.

- **Arquitectura del Modelo (`wed_night.py`)**:
  - **Encoder Convolucional**: Se utiliza una red convolucional de 4 capas para procesar las matrices de conectividad. Las capas convolucionales son ideales para capturar patrones espaciales locales en las matrices, que corresponden a subredes de conectividad. La arquitectura específica (`kernels: [7, 5, 5, 3]`, `strides: [2, 2, 2, 2]`) fue elegida para reducir progresivamente la dimensionalidad espacial.
  - **Espacio Latente**: Se define un espacio latente de `latent_dim = 512`. Esta dimensión es un hiperparámetro clave que equilibra la compresión de la información con la preservación de características discriminativas.
  - **Decoder "ConvTranspose"**: Se utiliza un decoder basado en convoluciones transpuestas para reconstruir la matriz de conectividad original desde el espacio latente, asegurando la simetría en la arquitectura.
  - **Regularización y Optimización**:
    - **Annealing Cíclico de Beta (β)**: En lugar de un valor β fijo, se emplea un annealing cíclico (`cyclical_beta_n_cycles = 2`). Esta técnica permite que el modelo se enfoque inicialmente en la reconstrucción (β bajo) y luego, de manera progresiva y periódica, en la regularización del espacio latente (β alto). Esto previene el colapso del término KL y promueve un espacio latente más informativo.
    - **Función de Activación `tanh`**: Se elige `tanh` como la activación final del decoder, adecuada para datos normalizados con z-score que están centrados en cero.
    - **Capa `LayerNorm`**: Se utiliza `LayerNorm` en las capas fully-connected para estabilizar el entrenamiento.

### 3. Clasificación Supervisada

Las representaciones latentes (`mu`, la media de la distribución latente) extraídas por el VAE se utilizan como características para la clasificación.

- **Validación Cruzada Anidada (Nested Cross-Validation)**:
  - **Bucle Externo (5 folds)**: Se utiliza para proporcionar una estimación insesgada del rendimiento del modelo final.
  - **Bucle Interno (5 folds)**: Se emplea dentro de cada fold externo para la optimización de hiperparámetros de los clasificadores a través de `GridSearchCV`.
  - **Estratificación**: La división de los datos se estratifica por `Sex` y `ResearchGroup` para asegurar que la distribución de estas variables sea consistente en los conjuntos de entrenamiento y prueba, reduciendo el riesgo de sesgos.

- **Clasificadores**: Se evalúa un conjunto diverso de modelos para determinar el más adecuado para el espacio latente aprendido:
  - Random Forest (`rf`)
  - Gradient Boosting (`gb`)
  - Support Vector Machine (`svm`)
  - Regresión Logística (`logreg`)
  - Perceptrón Multicapa (`mlp`)

- **Métrica de Optimización**: `balanced_accuracy` se utiliza como la métrica para el `GridSearchCV`. Esta es una elección fundamental en datasets médicos que pueden tener un ligero desequilibrio de clases, asegurando que el modelo se optimice para un buen rendimiento en ambas clases.

---

## Resultados Destacados

La siguiente tabla resume el rendimiento promedio de los clasificadores en los 5 folds de validación cruzada externa.

| Clasificador         | AUC    | Balanced Accuracy | F1-Score |
| -------------------- | ------ | ----------------- | -------- |
| **Random Forest (rf)** | **0.9240** | **0.8363** | **0.8500** |
| Gradient Boosting (gb) | 0.5863 | 0.5161            | 0.4706   |
| SVM (svm)            | 0.8933 | 0.7822            | 0.8000   |
| Logistic Regression (logreg)  | 0.9064 | 0.8363            | 0.8500   |
| MLP (mlp)            | 0.8626 | 0.8333            | 0.8636   |

**Análisis de Resultados**: El clasificador **Random Forest** demuestra el mejor rendimiento general, alcanzando un AUC de 0.9240 y una precisión balanceada de 0.8363, lo que indica una excelente capacidad para generalizar y clasificar correctamente a los sujetos. La Regresión Logística y el MLP también muestran un rendimiento muy competitivo, sugiriendo que el espacio latente aprendido por el VAE es altamente linealmente separable y rico en características discriminativas.

---

## Cómo Ejecutar el Pipeline

1.  **Configuración del Entorno**: Asegúrate de tener Python 3.8+ y las librerías listadas en `requirements.txt` instaladas.

2.  **Ejecución del Script**: Ejecuta el pipeline desde la terminal utilizando el script `wed_night.py`. A continuación se muestra un ejemplo de comando basado en la ejecución documentada:

    ```bash
    python wed_night.py \
        --global_tensor_path /ruta/a/tu/GLOBAL_TENSOR.npz \
        --metadata_path /ruta/a/tu/SubjectsData.csv \
        --channels_to_use 1 2 3 \
        --outer_folds 5 \
        --inner_folds 5 \
        --classifier_types rf gb svm logreg mlp \
        --latent_dim 512 \
        --lr_vae 1e-4 \
        --epochs_vae 550 \
        --beta_vae 1.0 \
        --cyclical_beta_n_cycles 2 \
        --save_fold_artefacts
    ```

3.  **Salidas**: Los resultados, incluyendo los modelos entrenados por fold, los scalers y los historiales de entrenamiento del VAE, se guardarán en el directorio especificado por `--output_dir`.

---

## Justificación Teórica y Discusión

La elección de un VAE convolucional se fundamenta en la hipótesis de que las alteraciones cerebrales en la EA no son aleatorias, sino que siguen patrones topológicos específicos. Las CNNs son expertas en detectar estas regularidades espaciales en las matrices de conectividad. La posterior regularización del espacio latente por parte del VAE asegura que las representaciones aprendidas no solo sean compactas, sino también semánticamente significativas, facilitando la tarea de los clasificadores posteriores. Los resultados obtenidos, con un AUC superior a 0.9, validan este enfoque, demostrando que las características latentes extraídas son biomarcadores potentes para la clasificación de la EA.
