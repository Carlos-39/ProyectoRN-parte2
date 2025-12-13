# ⭐ Yelp Review Classification with RNN, LSTM & GRU

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![GloVe](https://img.shields.io/badge/GloVe-4A90E2?logo=nlp&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

Este repositorio contiene la **segunda entrega del proyecto de Redes Neuronales** de la Universidad del Valle, desarrollado en equipo por **Carlos Corrales** y **José Manuel Palma**.  
Se avanza desde los MLP (primera entrega) hacia arquitecturas especializadas en **procesamiento de secuencias**: **RNN**, **LSTM** y **GRU**, todas en modo **bidireccional**, para clasificar reseñas de Yelp en **5 categorías (1–5 estrellas)**.

> 📄 **Informe técnico completo disponible en [`InformeRN-Corrales-Palma.pdf`](InformeRN-Corrales-Palma%20(1).pdf)**.

---

## 🎯 Objetivo

Comparar el rendimiento de tres arquitecturas recurrentes en un problema de **clasificación multiclase de texto**, integrando:
- **Embeddings preentrenados GloVe** para representación semántica rica.
- **Regularización** (L2, Dropout, recurrent_dropout).
- **Normalización por capa** (LayerNormalization).
- **Early stopping** para evitar sobreajuste.

---

## 🧪 Tecnologías y técnicas clave

| Componente | Uso |
|----------|-----|
| **GloVe (100d)** | Representación densa y semántica de palabras. |
| **RNN / LSTM / GRU** | Modelado de dependencias secuenciales en texto. |
| **Bidireccionalidad** | Captura contexto pasado y futuro en cada palabra. |
| **LayerNormalization** | Estabiliza el entrenamiento y acelera la convergencia. |
| **Orthogonal Initialization** | Mejora el flujo de gradientes en capas recurrentes. |
| **EarlyStopping + Adam** | Control de sobreajuste y optimización robusta. |

---

## 📁 Contenido del notebook

### 1. **Preprocesamiento mejorado**
- Mismo pipeline que la primera entrega (tokenización, padding).
- Integración de **GloVe 6B 100d**: construcción de matriz de embeddings de tamaño `10,000 × 100`.
- Capa `Embedding` con `trainable=False` para preservar conocimiento preentrenado.

### 2. **Arquitecturas implementadas**
Todas comparten la misma estructura base:

```python
[Embedding(GloVe) → Bidirectional(RNN/LSTM/GRU) → LayerNorm → Dense(64) → Dropout → Dense(5, softmax)]
[RNN simple: como baseline secuencial]
[LSTM: para manejar dependencias a largo plazo]
[GRU: versión más ligera y eficiente de la LSTM]
```
### 3. **Evaluación rigurosa**
- Matriz de confusión por clase (1–5 estrellas).
- Curvas de accuracy y loss en entrenamiento vs. validación.
- Análisis de sobreajuste y capacidad de generalización.
- Comparación cruzada entre arquitecturas.

---

## 📈 Hallazgos clave (resumen del informe)
- Las redes recurrentes superan claramente al MLP en tareas de texto secuencial.
- LSTM y GRU logran mejor captura de contexto que la RNN simple.
- El modelo sigue teniendo mayor precisión en clases 4 y 5 (reseñas positivas), reflejando el desequilibrio del dataset.
- A pesar del uso de técnicas avanzadas, todas las arquitecturas muestran sobreajuste, sugiriendo la necesidad de:
  - Más datos balanceados.
  - Enfoques de clasificación ordinal.
  - Arquitecturas más modernas (ej: Transformers).

---

## 📥 Cómo usar

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/Carlos-39/yelp-review-classification-rnn-lstm-gru.git

2. **Instala dependencias**:
   ```bash
   pip install tensorflow pandas matplotlib seaborn scikit-learn ydata-profiling

3. **Descarga GloVe (si no está incluido)**
   ```bash
   wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip

4. **Ejecuta el notebook**
   ```bash
   jupyter notebook ProyectoRN_2.ipynb

---

## 👥 Autores
- Carlos Daniel Corrales (2122878)
- José Manuel Palma (2125182)
- Estudiantes de Ingeniería de Sistemas en la Universidad del Valle (Cali, Colombia).
Este proyecto forma parte de la asignatura Redes Neuronales.

---

### 💡 Hecho por [Carlos Corrales](https://github.com/Carlos-39)

Estudiante de **Ingeniería de Sistemas** en la **Universidad del Valle (Cali, Colombia)**  
Apasionado por la **Inteligencia Artificial**, **Ciencia de Datos** y el **desarrollo colaborativo**.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white&style=flat)](https://www.linkedin.com/in/carlos-daniel-corrales)
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white&style=flat)](https://github.com/Carlos-39)
[![Email](https://img.shields.io/badge/Email-D14836?logo=gmail&logoColor=white&style=flat)](mailto:carlos.corrales.ar21@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-FF6F00?logo=vercel&logoColor=white&style=flat)](https://TU_PORTAFOLIO)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?logo=instagram&logoColor=white&style=flat)](https://instagram.com/carlosdca_)
