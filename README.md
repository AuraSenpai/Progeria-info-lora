# Progeria-info-lora

# 🧑‍⚕️ Nao Healthcare Dialogue Dataset  

This repository contains the **Nao Healthcare Dialogue Dataset**, a synthetic dataset for fine-tuning conversational AI models with **LLaMA Factory**.  
It is designed around a **NAO robot** acting as a friendly healthcare companion, answering patient questions in a **casual, supportive, and approachable style**.  

---

## 📊 Dataset Overview  
- **Samples**: ~1,500  
- **Intents**: 4 (`greetings`, `diagnosis`, `treatment`, `precautions`)  
- **Format**: JSONL (`nao_dataset.jsonl`)  
- **Style**: Instruction-tuning (prompt → completion)  

---

## 📂 Repository Contents  
- `nao_dataset.jsonl` → Main dataset for training  
- `generate_nao_dataset_jsonl.py` → Script to regenerate or customize dataset  
- `README.md` → Documentation  

---

## 🔧 How to Use with LLaMA Factory  

1. Clone this repository or download the dataset:  
   ```bash
   git clone https://github.com/yourusername/nao-healthcare-dataset.git
   cd nao-healthcare-dataset
