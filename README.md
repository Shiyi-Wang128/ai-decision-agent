# AI Decision Agent

A multi-tool AI agent that answers business questions about e-commerce data 
using natural language. Built with LLM-powered Text-to-SQL and XGBoost demand forecasting.

## Demo

**Query:** "哪些品类销量最高？" (Which product categories have the highest sales?)

The agent automatically routes the question to the right tool, generates SQL, 
queries the database, and returns results with business insights.

## Architecture
```
User Question (natural language)
↓
LLM Router (decides which tool to use)
↓
┌───────────────────┬──────────────────────┐
│   Text-to-SQL     │   Demand Forecast    │
│  (historical data)│  (future prediction) │
└───────────────────┴──────────────────────┘
↓
Result + Business Insight (in natural language)
```

## Features

- **Natural language interface**: Ask questions in plain language, no SQL knowledge needed
- **Automatic tool routing**: Agent decides whether to query database or run ML model
- **Text-to-SQL**: Converts natural language to PostgreSQL queries automatically
- **Demand forecasting**: XGBoost model predicts future sales by product category
- **Feature importance**: Shows which factors drive the prediction
- **Business insights**: LLM summarizes results into actionable recommendations

## Tech Stack

- **LLM**: OpenAI GPT-4o (routing, SQL generation, insight generation)
- **ML Model**: XGBoost with feature importance analysis
- **Database**: PostgreSQL
- **Backend**: Python, SQLAlchemy, FastAPI (in progress)
- **Frontend**: Streamlit
- **Data**: Olist Brazilian E-Commerce Dataset (100K+ orders)

## Project Structure
```
ai_decision_agent/
├── app/
│   ├── agent/
│   │   ├── core.py          # Agent routing and orchestration
│   │   └── tools/
│   │       ├── sql_tool.py  # Text-to-SQL tool
│   │       └── ml_tool.py   # XGBoost forecasting tool
├── data/
│   └── raw/                 # Olist dataset (not included)
├── models/                  # Trained XGBoost model
├── pipelines/
│   ├── build_dataset.py     # Load CSV data into PostgreSQL
│   └── train_model.py       # Train XGBoost demand forecast model
├── frontend/
│   └── app.py               # Streamlit UI
└── utils/
└── db.py                # Database connection
```

## Setup

1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-decision-agent.git
cd ai-decision-agent
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
cp .env.example .env
# Fill in your credentials
```

4. Load data into database
```bash
python -m pipelines.build_dataset
```

5. Train the model
```bash
python -m pipelines.train_model
```

6. Run the app
```bash
python -m streamlit run frontend/app.py
```

## Dataset

[Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 100K+ orders, 32K+ products, 71 product categories.

## Author

Shiyi Wang | MS Data Science, UC San Diego