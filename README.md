# PsyThera-LLM — Fine-Tuning, Preprocesamiento, Anonimización y Evaluación de Alucinaciones

**Juan Sebastián Ramírez Ayala**  
📚 Ingeniería Estadística — Escuela Colombiana de Ingeniería  

**Daniel Felipe Ruiz Bermúdez**  
📚 Ingeniería Estadística — Escuela Colombiana de Ingeniería  

## Descripción general del proyecto
Este proyecto implementa todo el pipeline completo para construir un LLM especializado en diálogo terapéutico.  
Incluye limpieza y anonimización del dataset, fine-tuning con LoRA, carga del modelo, interfaz CLI y evaluación del modelo evaluando alucinaciones.

## Estructura del repositorio


PsyThera-LLM/
│
├── .gitignore
│
├── cli/
│   └── CLI_interface.py
│
├── notebooks/
│   └── Copia_de_DLLM_FINPROJECT (1).ipynb
│
├── evaluation/
│   └── evaluacion_hallucinations_model.csv
│
├── requirements.txt
│
└── README.md

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

## 6. Conversión a HuggingFace Dataset
El dataset se transforma a formato HuggingFace e incorpora plantilla estilo instruct:


[USER] pregunta
[THERAPIST] respuesta


---
## 7. Tokenización
- Uso de tokenizer de TinyLlama-1.1B-Chat-v1.0
- Tokenización de preguntas y respuestas
- Padding y truncamiento a longitud máxima de 1024 tokens
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

## 10. Convertir el modelo entrenado en PyTorch a formato GGUF

Para habilitar la ejecución del modelo fine-tuneado dentro de Ollama, fue necesario transformar el checkpoint de HuggingFace (generado por `Trainer`) al formato **GGUF**, compatible con el motor `llama.cpp`. El procedimiento realizado fue el siguiente:

1. **Identificación del checkpoint final del modelo**
   - Una vez completado el entrenamiento, se seleccionó el checkpoint correspondiente al mejor desempeño.
   - Este checkpoint incluía los archivos esenciales:
     - `config.json`  
     - `model.safetensors`  
     - `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`  
   - El directorio del checkpoint tenía la estructura estándar de un modelo HuggingFace.

2. **Clonación del repositorio `llama.cpp`**
   - Para usar los scripts de conversión oficiales, se descargó el repositorio:
     ```bash
     cd D:/
     git clone https://github.com/ggerganov/llama.cpp
     cd llama.cpp
     ```
   - Este repositorio incluye `convert_hf_to_gguf.py`, necesario para convertir modelos HF a GGUF.

3. **Ejecución directa del script de conversión**
   - En este caso **no fue necesario crear un entorno virtual**, ya que el sistema ya tenía las dependencias requeridas (`transformers`, `safetensors`, etc.).
   - Por tanto, se procedió directamente a ejecutar la conversión:
     ```bash
     python convert_hf_to_gguf.py ^
       "D:/.../psych_tinyllama_L/checkpoint-876" ^
       --outfile "D:/psych_tinyllama_L.gguf" ^
       --outtype f16
     ```
   - Se seleccionó `f16` para obtener un modelo más liviano manteniendo buena calidad numérica.

4. **Confirmación de la exportación**
   - El proceso finalizó mostrando:
     ```
     Model successfully exported to D:\psych_tinyllama_L.gguf
     ```
   - Esto verificó que el modelo había sido convertido correctamente.

5. **Alojamiento del archivo `.gguf` dentro del directorio de Ollama**
   - Tras generar el archivo GGUF, este tuvo que ser **copiado manualmente** al directorio donde Ollama almacena sus modelos, el cual normalmente incluye las subcarpetas:
     ```
     C:\Users\<usuario>\.ollama\models\
     ```
   - Esto garantiza que Ollama pueda encontrar el modelo y usarlo.

6. **Creación del `Modelfile`**
   - Una vez colocado el `.gguf` dentro del directorio de modelos de Ollama, se creó un archivo llamado `Modelfile` en la misma ubicación.  
   - Ese archivo indica a Ollama:
     - cuál es el archivo base del modelo (el `.gguf`)
     - cuál es el template de interacción
     - y los parámetros de generación
   - Ejemplo del `Modelfile` utilizado:
     ```
     FROM ./psych_tinyllama_L.gguf

     TEMPLATE """
     <|system|>
     You are a helpful therapist AI. Act kindly and professionally.

     <|user|>
     {{ .Prompt }}

     <|assistant|>
     """

     PARAMETER temperature 0.7
     PARAMETER top_p 0.9
     ```

7. **Construcción final del modelo en Ollama**
   - Con el `Modelfile` listo, se ejecutó:
     ```bash
     ollama create psych-therapist -f Modelfile
     ```
   - Esto generó un modelo instalable y ejecutable en Ollama bajo el nombre `psych-therapist`.

---

En síntesis, el proceso completa la transformación desde un checkpoint de PyTorch a un modelo `.gguf` compatible con Ollama, alojándolo en el directorio correspondiente y configurando su comportamiento mediante un `Modelfile`.


---

## 11. Interfaz CLI
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



