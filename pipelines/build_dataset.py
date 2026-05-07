import pandas as pd
import os
from sqlalchemy import text
from utils.db import get_engine

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)

def build():
    engine = get_engine()

    tables = {
        'customers': 'olist_customers_dataset.csv',
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'order_payments': 'olist_order_payments_dataset.csv',
        'order_reviews': 'olist_order_reviews_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'category_translation': 'product_category_name_translation.csv',
    }

    for table_name, filename in tables.items():
        print(f"Loading {filename} → table: {table_name}")
        df = load_csv(filename)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"  Done: {len(df)} rows")

if __name__ == '__main__':
    build()