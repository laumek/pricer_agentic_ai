# PRICER agentic AI

## Autonomous Multi-Agent Deal Hunter Using LLMs, RAG, Vector Databases, and Hybrid ML

This project builds a fully autonomous AI system that scans the web for deals, estimates the *true* market price using an ensemble of models,   and sends real-time push notifications when something is genuinely undervalued.
It combines **LLMs, classical ML, vector search, fine-tuning, cloud GPUs, Gradio-based UI, and agentic workflows** into a cohesive end-to-end system.

## Overview
This project builds a fully autonomous AI system that scans the web for deals, estimates the *true* market price using an ensemble of models,   and sends real-time push notifications when something is genuinely undervalued.
It combines **LLMs, classical ML, vector search, fine-tuning, cloud GPUs, Gradio-based UI, and agentic workflows** into a cohesive end-to-end system.



## ⚙️ System Architecture
1. EnsembleAgent — Aggregates multiple pricing models:
* SpecialistAgent: Fine-tuned LLM (QLoRA) deployed on Modal.
* FrontierAgent: Retrieval-Augmented Generation (RAG) model using GPT-4o-mini / DeepSeek.
* RandomForestAgent: Traditional ML model trained on sentence-transforme embeddings.
2. ScannerAgent — RSS feed scraper that collects real-time deal listings.
3. MessagingAgent — Push notification service for high-value opportunities.
4. PlanningAgent — Central orchestrator that manages agent workflows and decision logic.
5. Gradio UI — Interactive dashboard for viewing deals, model estimates, and alerts.

<img width="637" height="312" alt="Screenshot 2025-12-03 at 15 53 19" src="https://github.com/user-attachments/assets/4941d16f-737b-46ea-b272-38b794a77cab" />
<img width="661" height="275" alt="Random Fort" src="https://github.com/user-attachments/assets/c8b1e099-16e2-4283-95b0-59db592ae409" />


## 🧩 Data Pipeline
1. Data Collection
* Curated Pricing Dataset (Hugging Face): Loaded Amazon product metadata from McAuley-Lab/Amazon-Reviews-2023 across 8 categories: Automotive, Electronics, Office Products, Tools & Home Improvement, Cell Phones & Accessories, Toys & Games, Appliances, Musical Instruments. These entries provide product descriptions + prices used to train all pricing models.
* Live Deal Scraping: RSS feeds (e.g., SlickDeals, HotUKDeals) supply real-time deal descriptions and prices for inference.

2. Data Cleaning & Transformation
* Normalised product descriptions (titles + bullet points → clean text)
* Extracted & validated pricing information
* Removed duplicates and outliers
* Result: a consistent price–description dataset suitable for model training and evaluation.

3. Embeddings & Storage
* Embedded all product descriptions using sentence-transformers/all-MiniLM-L6-v2
* Stored vectors + metadata in ChromaDB → Enables similarity search for the frontier RAG model → Provides neighbourhood price statistics (min/max) used in the ensemble

4. Model Training
* Specialist LLM: fine-tuned with QLoRA on curated dataset for price prediction
* Frontier RAG Model: retrieves nearest embeddings → frontier LLM estimates fair value
* Random Forest Baseline: trained on embeddings to provide a stable numeric estimate
* Combined through a calibrated linear ensemble, using real learned coefficients.

5. Real-Time Deal Scoring
For every incoming deal:
1. Embed description
2. Retrieve similar items from ChromaDB
3. Generate three independent predictions
4. Combine via ensemble to compute fair market value
5. Compare against scraped price to compute discount
6. If discount exceeds threshold → push notification

## 🧮 Features

* Autonomous Deal Scanning: Constantly monitors RSS feeds for new listings.
* Fair Price Estimation: Ensemble of LLM and ML models to infer realistic market value.
* Push Notifications: Alerts when significant undervaluations are detected.
* Gradio Dashboard: Clean UI for visualization, manual validation, and control.
* Multi-Agent Reasoning: Modular design with orchestrated agent collaboration.

### Multi-Agent Architecture  

| Agent | Purpose |
|-------|---------|
| **Scanner Agent** | Scrapes RSS feeds for new deals |
| **Frontier Agent (RAG)** | Retrieves similar items via embeddings + uses frontier LLM to estimate price |
| **Specialist Agent (Fine-Tuned LLM)** | QLoRA fine-tuned model predicts clean prices |
| **Random Forest Agent** | Predicts price using sentence-transformer embeddings |
| **Ensemble Agent** | Linear model combining all price predictions |
| **Planning Agent** | Picks best deal, calculates discount, triggers alerts |
| **Messaging Agent** | Sends Pushover alerts or SMS notifications |

## Ensemble Model (Meta-Model)

The system doesn't rely on one model.  
It **learns** how to weight them optimally using a trained linear regression:
FinalPrice = 0.73 * SpecialistLLM +1 .03 * FrontierLLM + 0.44 * RandomForest + 0.64 * MinModel + 0.60 * MaxModel - 26.47

## Specialist Model (Fine-Tuned LLM)

- QLoRA fine-tuned on ~400k product descriptions  
- Runs in 4-bit quantized mode  
- Deployed to **Modal** as a GPU-backed inference service  
- Stateless, fast cold starts, cached weights  
![IMG_0187](https://github.com/user-attachments/assets/e99ffc12-d182-4c1a-a519-553b452d3981)
<img width="661" height="275" alt="Random Fort" src="https://github.com/user-attachments/assets/9e009a4d-fe3a-46c8-8f5c-8bd4d1108a74" />


## 📡 Modal Deployment

The specialist model is exposed via:

```python
Pricer = modal.Cls.lookup("pricer-service", "Pricer")
pricer.price.remote("product description")
```



## 🧱 Tech Stack
| Layer                   | Technology                               |
|-------------------------|-------------------------------------------|
| Language                | Python                                    |
| LLM & Fine-tuning       | QLoRA, Transformers, PEFT                 |
| Embeddings              | SentenceTransformers / OpenAI embeddings  |
| Vector DB               | ChromaDB                                  |
| Frontend / UI           | Gradio                                    |
| Agents / Orchestration  | LangChain / custom planning logic         |
| Notifications           | Pushover.net API                          |
| Deployment              | Modal (GPU service) / Localhost           |


## 🖥️ Gradio Monitoring Dashboard
The UI includes:
* real-time agent logs
* a table of all discovered deals
* click-to-alert functionality
* 3D embedding visualization (vector DB)
* automatic refresh (300s)
![The Price is laght - Autonomous Agent Framework that hurts lor deals](https://github.com/user-attachments/assets/44d9d9b3-0953-48fc-a6e4-2975816a39df)


## 🔧 Getting Started to run locally

1. Clone the repository
```git clone https://github.com/laumek/pricer_agentic_ai.git```

```cd pricer_agentic_ai```

2. Install dependencies from pyproject.toml file
```pip install -e .```

4. Set up environment variables

Use .env.example to create a .env file with your API keys and configuration:
``` OPENAI_API_KEY=...
HF_TOKEN=...
PUSHOVER_USER=...
PUSHOVER_TOKEN=...
etc.
```
4. Run the system
```python src/price_intel/agents/main.py```
5. Launch the Gradio UI
```python src/price_intel/interface/gradio_app.py```

## 🙌 Acknowledgements
* Hugging Face datasets for curated product data.
* SentenceTransformers for embeddings.
* Modal for deployment and LLM inference.
* Gradio for rapid UI prototyping.
* OpenAI / DeepSeek models for RAG and reasoning layers.
* This project builds on code from Ed Donner (https://github.com/ed-donner/llm_engineering) under the MIT License. Significant modifications, enhancements, and additional agents have been implemented independently.
