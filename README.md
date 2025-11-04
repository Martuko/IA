# Tarea 2 — IA UDP (Proyecto Base)

Este repo contiene el esqueleto listo para ejecutar en VS Code/Jupyter.

## Estructura
```
t2_ai_udp/
├── config/
│   └── config.yaml
├── data/
│   └── ai4i_2020.csv        # <-- Coloca aquí tu archivo descargado de Kaggle
├── notebooks/
│   └── T2_AI_UDP.ipynb
├── src/
│   ├── clustering.py
│   ├── supervised.py
│   └── utils.py
└── requirements.txt
```

## Pasos
1. (Descarga) Coloca el CSV en `data/` y actualiza `config/config.yaml` con:
   - `dataset_path`: ruta al archivo CSV.
   - `target_column`: nombre de la etiqueta Y (debe ser **discreta** con **>2** clases).
   - `drop_columns` / `categorical_columns` si aplica.
2. (Instala) `pip install -r requirements.txt`
3. (Abre) `notebooks/T2_AI_UDP.ipynb` en VS Code y ejecuta secuencialmente.

> Recuerda: debes redactar tu **propio análisis**. El uso de herramientas generativas para redactar análisis está **prohibido por la pauta**.
