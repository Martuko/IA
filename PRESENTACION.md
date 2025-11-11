# Tarea 2 - Inteligencia Artificial UDP
## Clustering y Modelos Supervisados con Eliminación Competitiva

---

## 1. Dataset: AI4I 2020 Predictive Maintenance

### Descripción General
- **Fuente**: Kaggle - AI4I 2020 Predictive Maintenance Dataset
- **Tamaño**: 10,000 observaciones
- **Objetivo**: Clasificación de tipos de falla en máquinas industriales
- **Variable objetivo**: Failure Type (multiclase)

### Características del Dataset
**Variables numéricas (6):**
- Air temperature [K]: Temperatura del aire
- Process temperature [K]: Temperatura del proceso
- Rotational speed [rpm]: Velocidad rotacional
- Torque [Nm]: Torque aplicado
- Tool wear [min]: Desgaste de herramienta

**Variables categóricas (1):**
- Type: Tipo de producto (L/M/H - Low/Medium/High quality)

**Variables de falla (eliminadas):**
- Machine failure (binaria general)
- TWF, HDF, PWF, OSF, RNF (tipos específicos de falla - redundantes)

### Distribución de Clases
El análisis reveló un **desbalance severo**:
- Clase mayoritaria: ~96.5% (No Failure)
- Clase minoritaria: ~3.5% (Failure)

Este desbalance tiene implicaciones críticas en la evaluación de modelos.

---

## 2. Clustering: Aprendizaje No Supervisado

### Metodología

**Técnicas evaluadas:**
1. **K-Means** (inicialización random)
2. **K-Means++** (inicialización inteligente)
3. **MeanShift** (determinación automática de clusters)

**Configuraciones:**
- ≥4 configuraciones por técnica
- Variación de K (2, 3, 4, 5, 6, 8)
- Variación de quantile para MeanShift (0.10, 0.20, 0.30)
- MeanShift con bandwidth fijo (1.0)

### Proceso Experimental

**División de datos:**
- 80% training (sin etiquetas Y)
- 20% test (con etiquetas Y para validación)

**Métrica de evaluación:**
- Silhouette Score en train
- Asignación por clase dominante en test

**Pipeline:**
1. Preprocesamiento: StandardScaler para numéricas, OneHotEncoder para categóricas
2. Entrenamiento de múltiples configuraciones en train
3. Selección de TOP-3 según Silhouette
4. Aplicación al test y cálculo de precisión por etiqueta dominante

### Resultados TOP-3

| Rank | Técnica | Config | K | Silhouette | Test Acc |
|------|---------|--------|---|------------|----------|
| 1 | MeanShift | ms_q_0.10 | auto | 0.390 | 96.6% |
| 2 | K-Means | km_rand_k2 | 2 | 0.229 | 96.5% |
| 3 | K-Means++ | km_pp_k3 | 3 | 0.228 | 96.5% |

### Análisis de Resultados

**MeanShift (ganador):**
- Mejor separación interna (Silhouette = 0.390)
- No requiere especificar K manualmente
- Determinación automática vía bandwidth=1.895
- Excelente generalización: 96.6% de precisión

**K-Means con K=2:**
- Silhouette moderado-bajo (0.229)
- Sin embargo, alta precisión en test (96.5%)
- Indica estructura binaria natural del dataset
- K=2 captura la división Failure/No-Failure

**K-Means++ con K=3:**
- Rendimiento idéntico a K=2
- Subdivisión adicional no aporta información discriminativa
- Confirma que K=2 es óptimo

### Interpretación

**Consistencia train-test excepcional:**
Las tres configuraciones logran ~96.5-96.6% de precisión, indicando que:
- Los clusters son representativos de clases reales
- Las features discriminan efectivamente
- Clustering no supervisado redescubre la estructura de clases

**Paradoja Silhouette vs. Precisión:**
MeanShift tiene Silhouette 70% mayor que K-Means, pero la diferencia en precisión es mínima (0.1%). Esto sugiere:
- Silhouette bajo puede deberse a solapamiento natural entre clases
- La métrica de "clase dominante" es más robusta para evaluar utilidad práctica

**Número óptimo de clusters:**
K=2 emerge como el número natural, confirmando la estructura binaria del problema (Failure vs No-Failure).

---

## 3. Modelos Supervisados: Eliminación Competitiva

### Metodología

**Algoritmos:**
1. **Regresión Logística** (SGDClassifier con loss='log_loss')
2. **SVM** (SGDClassifier con loss='hinge' y 'modified_huber')

**Hiperparámetros explorados:**
- Alpha (regularización): 5e-5, 1e-4, 5e-4
- Learning rate: 'optimal', 'adaptive'
- Eta0 (adaptive): 0.01, 0.05

**Proceso de eliminación:**
- 50 épocas totales
- Evaluación cada 5 épocas
- Eliminación de la peor instancia cada evaluación
- Sobreviven las 2 mejores configuraciones

**Métricas:**
- F1-macro (métrica de selección)
- Accuracy
- F1 por clase

### Configuraciones Evaluadas

**Logistic Regression (4 configs):**
- log_optimal_a1e-4
- log_optimal_a5e-5 ✓ (sobreviviente)
- log_adapt_eta005_a5e-4
- log_adapt_eta001_a1e-4

**SVM (4 configs):**
- svm_modhub_opt_a1e-4 ✓ (sobreviviente)
- svm_hinge_opt_a1e-4
- svm_modhub_adp_eta005_a5e-4
- svm_hinge_adp_eta001_a1e-4

### Resultados Finales

| Modelo | Config | Test Accuracy | F1-macro |
|--------|--------|---------------|----------|
| SVM | svm_modhub_opt_a1e-4 | **96.75%** | 0.246 |
| Logistic | log_optimal_a5e-5 | 44.0% | 0.106 |

### Análisis Crítico

**Problema 1: Discrepancia Accuracy vs F1**

El SVM alcanza 96.75% de accuracy pero solo 0.246 de F1-macro. Esta discrepancia revela:
- El modelo predice casi siempre la clase mayoritaria (No-Failure)
- Accuracy es engañoso en datasets desbalanceados
- El modelo **no está aprendiendo** a detectar fallas realmente

**Problema 2: Sobre-regularización**

Ambos modelos usan valores de C (alpha inverso) extremadamente bajos:
- SVM: alpha=1e-4 → C=10,000
- Logistic: alpha=5e-5 → C=20,000

Valores típicos de C: 0.1 a 100. Esto causa **under-fitting** severo.

**Problema 3: Regresión Logística Fallida**

44% de accuracy es peor que el azar (50% para binario). Causas:
- Sobre-regularización extrema
- Modelo no convergió
- Features no linealmente separables sin transformación

**Problema 4: Proceso de Eliminación Defectuoso**

Si la eliminación se basa en accuracy en train:
- Se favorecen modelos que ignoran la clase minoritaria
- Los modelos "inteligentes" que balancean clases son eliminados temprano
- Sobreviven los modelos sesgados

---

## 4. Análisis Comparativo

### Clustering vs. Supervisado

| Aspecto | Clustering (MeanShift) | Supervisado (SVM) |
|---------|------------------------|-------------------|
| Precisión test | 96.6% | 96.75% |
| F1-macro | N/A | 0.246 |
| Complejidad | Baja | Alta |
| Interpretabilidad | Alta | Baja |
| Requiere labels | No | Sí |

**Paradoja principal:**
El método **no supervisado** (clustering) iguala al supervisado en precisión, siendo más simple e interpretable.

### Desbalance de Clases: El Villano Oculto

**Evidencia del desbalance:**
- SVM: 96.75% accuracy pero 0.246 F1-macro
- Clustering: 96.6% precisión (consistente con proporción mayoritaria)
- Distribución: ~96.5% No-Failure vs ~3.5% Failure

**Impacto en modelos:**
- Accuracy inflado artificialmente
- Modelos aprenden a predecir siempre la clase mayoritaria
- Clase minoritaria (fallas) ignorada completamente
- F1-macro revela la verdad

### Hiperparámetros: Rango Inadecuado

**Problema detectado:**
Los valores de alpha (regularización) están fuera del rango típico:
- Configuración actual: 5e-5 a 5e-4
- Rango recomendado: 0.001 a 1.0 (equivalente a C: 1 a 1000)

**Consecuencia:**
Under-fitting severo, especialmente en Logistic Regression.

---

## 5. Conclusiones

### Hallazgos Principales

1. **Clustering es sorprendentemente efectivo:**
   - MeanShift logra 96.6% de precisión sin supervisión
   - Redescubre la estructura de clases automáticamente
   - K=2 emerge como número natural de clusters

2. **Desbalance de clases domina el experimento:**
   - ~96.5% vs ~3.5% de distribución
   - Accuracy es métrica engañosa
   - F1-macro revela el verdadero desempeño

3. **Modelos supervisados mal configurados:**
   - Sobre-regularización extrema (alpha demasiado bajo)
   - Rango de hiperparámetros inadecuado
   - Proceso de eliminación favorece modelos sesgados

4. **Paradoja: No supervisado > Supervisado:**
   - Clustering iguala precisión con menor complejidad
   - Supervisados fallan por configuración deficiente
   - Indica que el problema es casi linealmente separable

### Lecciones Aprendidas

**Sobre métricas:**
- **Nunca usar accuracy como única métrica** en datos desbalanceados
- F1-macro, recall y matriz de confusión son esenciales
- Una métrica alta puede ocultar un modelo inútil

**Sobre hiperparámetros:**
- Validar rangos antes de experimentar
- Sobre-regularización puede ser peor que sobre-ajuste
- Grid search debe cubrir órdenes de magnitud

**Sobre desbalance:**
- Identificar desbalance antes de modelar
- Aplicar técnicas de balanceo desde el inicio
- class_weight='balanced' debería ser default

**Sobre clustering:**
- No subestimar métodos no supervisados
- Silhouette no es la única métrica relevante
- Validación con etiquetas externas es valiosa

### Recomendaciones para Mejora

**Inmediatas:**
1. Re-entrenar con `class_weight='balanced'`
2. Ampliar rango de alpha: [0.001, 0.01, 0.1, 1.0]
3. Usar F1-macro para eliminación, no accuracy
4. Reportar matriz de confusión

**A mediano plazo:**
5. Aplicar SMOTE o undersampling
6. Probar otros clasificadores (Random Forest, XGBoost)
7. Ingeniería de features enfocada en clase minoritaria
8. Validación cruzada estratificada

**Análisis avanzado:**
9. Análisis de curvas ROC/PR
10. Análisis de error por tipo de falla
11. Estudio de features más discriminativas
12. Ensemble de clustering + supervisado

### Impacto Práctico

En un contexto industrial de mantenimiento predictivo:

**Modelo actual (SVM 96.75% acc, 0.246 F1):**
- Detecta ~0% de fallas reales
- Genera sensación falsa de seguridad
- Pérdidas por fallas no detectadas

**Modelo ideal (después de mejoras):**
- Balance entre precisión y recall
- Detecta mayoría de fallas (alto recall)
- Minimiza falsos positivos (costos de mantenimiento)
- Valor económico real

### Conclusión Final

Este experimento demuestra que:

1. **La configuración importa más que el algoritmo**: Modelos supervisados potentes fallaron por mala configuración, mientras que clustering simple triunfó.

2. **Las métricas deben alinearse con el objetivo**: Accuracy puede dar 96.75% mientras el modelo es inútil para detectar fallas.

3. **El desbalance de clases requiere tratamiento especial**: Ignorarlo convierte cualquier experimento en un ejercicio de futilidad.

4. **La simplicidad tiene valor**: MeanShift (no supervisado, sin hiperparámetros críticos) es actualmente la mejor opción para este problema.

**Recomendación final:** Usar MeanShift para producción mientras se re-entrena supervisado con configuración corregida.

---

## Apéndice: Código y Reproducibilidad

### Estructura del Proyecto
```
IA/
├── config/
│   └── config.yaml          # Configuración centralizada
├── data/
│   └── ai4i2020.csv         # Dataset original
├── notebooks/
│   ├── T2_AI_UDP.ipynb      # Notebook principal
│   └── outputs/
│       ├── train_history.csv
│       └── test_summary.csv
├── src/
│   ├── clustering.py        # run_clustering()
│   ├── supervised.py        # train_with_elimination()
│   └── utils.py             # Funciones auxiliares
├── requirements.txt
└── README.md
```

### Tecnologías Utilizadas
- Python 3.10+
- scikit-learn: clustering y modelos supervisados
- pandas: manipulación de datos
- NumPy: operaciones numéricas
- YAML: configuración externa

### Reproducibilidad
- Seed fijada: `random_state=42`
- Configuración versionada en YAML
- Pipeline determinista de preprocesamiento
- Resultados guardados en CSV

---

## Referencias

- Dataset: [AI4I 2020 Predictive Maintenance Dataset - Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- Documentación scikit-learn: Clustering, SGDClassifier
- Material del curso: Inteligencia Artificial UDP

---

**Fecha:** Noviembre 2025  
**Autor:** [Tu nombre]  
**Curso:** Inteligencia Artificial - Universidad Diego Portales
