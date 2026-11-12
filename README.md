# Proyecto BERT: Clasificación de sentimiento en IMDB

**Curso:** Aprendizaje Profundo Aplicado
**Autor:** Turpo Quispe Patty Milagros
**Fecha:** 12 de noviembre de 2026

Fine-tuning de `bert-base-uncased` (BERT-Base: 12 capas, 768 hidden, 12 cabezas, ~110 M de parámetros) para clasificar reseñas de películas de IMDB como **POSITIVE** o **NEGATIVE**, usando Hugging Face Transformers y PyTorch.

## Contenido

| Archivo | Descripción |
|---|---|
| `BERT_Sentimiento_IMDB.ipynb` | Notebook de Google Colab con todo el proyecto |
| `requirements.txt` | Dependencias (para ejecutar fuera de Colab) |
| `Informe_BERT.docx` | Informe con teoría, implementación, resultados y figuras |
| `figuras.zip` | Figuras generadas por el notebook |

## Cómo ejecutarlo en Google Colab

1. Entra a https://colab.research.google.com y sube el notebook (*Archivo → Subir cuaderno*).
2. Activa la GPU: *Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)*.
3. Ejecuta todo: *Entorno de ejecución → Ejecutar todo* (≈6 min de entrenamiento con la configuración por defecto).
4. Al final se guarda el modelo en `bert_sentiment_model/` y se genera `figuras.zip`.

### Ejecución local (opcional, requiere GPU)

```
pip install -r requirements.txt
jupyter notebook BERT_Sentimiento_IMDB.ipynb
```

## Qué hace el notebook

1. Instala dependencias y verifica la GPU; fija semillas (42) para reproducibilidad.
2. Carga IMDB y hace análisis exploratorio (balance de clases, longitud de reseñas).
3. Muestra la tokenización WordPiece (`[CLS]`, `[SEP]`, `token_type_ids`, `attention_mask`).
4. Entrena el modelo con `Trainer` y grafica las curvas de pérdida y métricas.
5. Evalúa: classification report, matriz de confusión, errores y predicciones de ejemplo.
6. Embeddings contextuales: similitud coseno de "apple" en tres contextos (polisemia).
7. Evaluación avanzada: curvas ROC y Precision-Recall, distribución de confianza, mapa de atención (capa 12) y proyección PCA del token `[CLS]`.
8. Guarda el modelo y comprime las figuras.

## Configuración

| Parámetro | Valor |
|---|---|
| Datos | 8 000 entrenamiento / 2 000 prueba (submuestreo de IMDB, semilla 42) |
| Longitud máxima | 256 tokens |
| Optimizador | AdamW, lr 2e-5, weight decay 0.01, warmup 150 pasos |
| Batch / épocas | 16 / 3 (1 500 pasos) |
| Precisión | fp16 en GPU |

Para mejores resultados puedes subir `N_TRAIN` a 25 000 en la tercera celda de código.

## Resultados (conjunto de prueba, GPU Tesla T4)

| Métrica | Valor |
|---|---|
| Accuracy | 0.908 |
| Precision (POSITIVE) | 0.897 |
| Recall (POSITIVE) | 0.922 |
| F1 (POSITIVE) | 0.909 |
| AUC (ROC) | 0.974 |
| Tiempo de entrenamiento | 332 s |

Similitud coseno de "apple": 0.75 entre dos usos como fruta, 0.45–0.47 frente al uso como empresa.

## Uso del modelo entrenado

```python
from transformers import pipeline
clf = pipeline("text-classification", model="bert_sentiment_model")
print(clf("I loved this film"))
```

## Limitaciones

- Se usa un subconjunto de IMDB; la pérdida de evaluación sube en la tercera época (ligero sobreajuste).
- Falla con sarcasmo y opiniones mixtas, y lo hace con alta confianza.
- Las reseñas de más de 256 tokens se truncan.

## Referencias

- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT.
- Alammar, J. *The Illustrated BERT.* http://jalammar.github.io/illustrated-bert/
- Hugging Face Transformers: https://github.com/huggingface/transformers
