# Samsung Innovation Campus - Proyecto de Análisis de Infecciones Respiratorias Agudas

## Descripción del Proyecto
Este proyecto analiza datos de vigilancia epidemiológica de Infecciones Respiratorias Agudas (IRA) en Colombia, desarrollado como parte del programa Samsung Innovation Campus.

## Estructura del Proyecto

### 📊 `notebooks/`

#### `EDA_inicial/`
Contiene el análisis exploratorio de datos (EDA) desarrollado durante los avances 1 y 2 del proyecto:
- `NOTEBOOK___Vigilancia_epidemiologica_de_Infecciones_Respiratoiras_Agudas_(IRA)_.ipynb`
- `NOTEBOOK___Morbilidad_Infecciones_respiratorias_agudas_altas.ipynb`

#### `EDA_final/`
Contiene el análisis exploratorio de datos definitivo utilizado para la presentación final:
- `1.0-EDA.ipynb` - EDA consolidado y refinado para la entrega final

#### `feature_engineering/`
Notebooks dedicados a la creación y transformación de variables:
- `create_df_cleaned.ipynb` - Limpieza y preparación inicial de datos
- `take_cleaned_to_create_casos.ipynb` - Creación de dataset de casos
- `2.0-create_sub_reg_nt.ipynb` - Creación de variables de subregión
- `2.1-create_consolidado.ipynb` - Consolidación de datos
- `2.2-create-casos-with-lags.ipynb` - Creación de variables con rezagos temporales

#### `feature_selection/`
Notebooks para selección de características relevantes:
- `select_features.ipynb` - Proceso de selección de features para el modelo

#### `modeling/`
Contiene los modelos de Machine Learning desarrollados por los diferentes integrantes del equipo (identificados por sus iniciales):
- `1.0-lq-model training.ipynb` - Modelo desarrollado por LQ
- `1.0-af-model training.ipynb` - Modelo desarrollado por AF
- `1.0-wv-model training.ipynb` - Modelo desarrollado por WV
- `1.0-js-model training.ipynb` - Modelo desarrollado por JS
- `1.0-or-model training.ipynb` - Modelo desarrollado por OR
- `final_model.ipynb` - Modelo final consolidado del proyecto
- `train_and_predict.py` - Script de entrenamiento y predicción

#### `envio_datos_power_bi/`
Prueba de concepto para integración con Power BI:
- `prueba_concepto.ipynb` - Implementación del envío de datos en tiempo real a Power BI

#### `data/`
Carpeta que contiene los datos del proyecto:
- `01_data_cruda/download_data.ipynb` - Notebook para descarga de datos originales

#### `reports/`
Carpeta destinada a almacenar reportes y documentación del proyecto

## Equipo
- LQ - Luiz Ignacio Quineche Casana
- AF - Angeli Flores Quito
- WV - Wisner Ernan Valdiviezo Goicochea
- JS - Jessica Amapola Serpa Buitrón
- OR - Oscar Romero Mayta

## Tecnologías Utilizadas
- Python
- Jupyter Notebooks
- Machine Learning (CatBoost, entre otros)
- Power BI (integración)

## Instalación y Uso
```bash
# Clonar el repositorio
git clone https://github.com/[usuario]/Samsung_innovation_campus_proyecto.git

# Instalar dependencias
pip install -r requirements.txt
```

## Contribuciones
Este proyecto fue desarrollado como parte del programa Samsung Innovation Campus.