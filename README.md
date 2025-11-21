# PsyThera-LLM — Fine-Tuning, Preprocesamiento, Anonimización y Evaluación de Alucinaciones

## Descripción general del proyecto
Este proyecto implementa todo el pipeline completo para construir un LLM especializado en diálogo terapéutico.  
Incluye limpieza y anonimización del dataset, fine-tuning con LoRA, carga del modelo, interfaz CLI y evaluación del modelo evaluando alucinaciones.

---

## 1. Carga del dataset
- Se utiliza KaggleHub para cargar el dataset Therapist Q&A Dataset.
- Exploración inicial del dataframe: formas, columnas y valores nulos.

---

## 2. Limpieza del dataset
Se realizan las siguientes acciones:
- Eliminación de preguntas y respuestas vacías.
- Eliminación de duplicados exactos.
- Limpieza de espacios en blanco y saltos de línea.
- Normalización de strings.

---

## 3. Anonimización (Reglas + NER)

### A. Anonimización por patrones (RegEx)
Se reemplazan automáticamente:
- URLs → [URL]
- Correos → [EMAIL]
- Teléfonos → [PHONE]
- Handles de redes → [HANDLE]

### B. Anonimización usando NER (spaCy)
Detección de entidades:
- PERSON → [PERSON]
- GPE → [GPE]
- LOC → [LOC]
- ORG → [ORG]

Se generan columnas anonimizadas (Question_anon, Answer_anon).

---

## 4. Detección de contenido sensible
Se identifican contenidos con riesgo mediante listas de palabras clave:
- Autolesión / suicidio
- Violencia
- Abuso sexual

Se asigna un nivel de riesgo:
- high
- medium
- none

Y se separan datasets: psych_safe.csv y psych_sensitive.csv.

---

## 5. Filtro por longitud
Se aplican filtros para remover ejemplos demasiado cortos o largos:
- Pregunta: 5–400 palabras
- Respuesta: 5–600 palabras

Resultado final: psych_clean_final.csv.

---

## 6. Normalización + creación JSONL
Se normalizan textos:


instruction: pregunta
response: respuesta


Se generan:
- psych_train.jsonl
- psych_test.jsonl
- psych_eval.jsonl

---

## 7. Conversión a HuggingFace Dataset
El dataset se transforma a formato HuggingFace e incorpora plantilla estilo instruct:


[USER] pregunta
[THERAPIST] respuesta


---

## 8. Fine-Tuning del modelo

### A. Phi-3-mini-4k-instruct con LoRA + QLoRA
- Cuantización 4-bit (BitsAndBytes)
- Adaptadores LoRA colocados en capas del modelo
- Entrenamiento con Trainer y guardado de checkpoints

### B. TinyLlama-1.1B-Chat-v1.0
Incluye:
- Collator especializado para entrenar solo la parte de [THERAPIST]
- Tokenización optimizada
- Entrenamiento completo del modelo

---

## 9. Guardado y uso del modelo finetuneado
- El mejor checkpoint se guarda automáticamente.
- Se prepara para uso en Ollama o por medio del pipeline de HuggingFace.

---

## 10. Interfaz CLI
Se incluye función generadora de prompts:


[USER] <pregunta>
[THERAPIST]


Y se habilita inferencia desde la línea de comandos con el modelo entrenado.

---

## 11. Evaluación de alucinaciones
Se evalúa el modelo usando 4 tipos de preguntas:
- Psicología (esperadas)
- Fuera de contexto (para detectar alucinación)
- Sensibles éticamente
- Control general

Se genera archivo final:
- evaluacion_hallucinations_model.csv

Cada fila contiene:
- Tipo de pregunta
- Pregunta
- Respuesta del modelo
- Columna para calificación manual

---

## Estructura sugerida del repositorio


PsyThera-LLM/
│
├── data/
│   ├── psych_clean_final.csv
│   ├── psych_train.jsonl
│   ├── psych_test.jsonl
│   └── psych_eval.jsonl
│
├── model/
│   └── checkpoint-292/
│
├── evaluation/
│   └── evaluacion_hallucinations_model.csv
│
├── cli/
│   └── cli_inference.py
│
└── README.md


---