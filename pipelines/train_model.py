import pandas as pd
import numpy as np
import pickle
import os
from sqlalchemy import text
from utils.db import get_engine

engine = get_engine()


def load_data():
    query = """
    SELECT 
        DATE_TRUNC('month', o.order_purchase_timestamp::timestamp) AS month,
        p.product_category_name AS category,
        COUNT(oi.order_item_id) AS sales_count
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE p.product_category_name IS NOT NULL
    AND o.order_purchase_timestamp IS NOT NULL
    GROUP BY month, category
    ORDER BY month, category
    """
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        columns = list(result.keys())
    return pd.DataFrame(rows, columns=columns)


def build_features(df):
    df['month'] = pd.to_datetime(df['month'])
    df = df.sort_values(['category', 'month'])

    # 时间特征
    df['month_num'] = df['month'].dt.month
    df['year'] = df['month'].dt.year

    # 滞后特征：上个月、上上个月的销量
    df['lag_1'] = df.groupby('category')['sales_count'].shift(1)
    df['lag_2'] = df.groupby('category')['sales_count'].shift(2)

    # 滚动平均：过去3个月平均销量
    df['rolling_mean_3'] = df.groupby('category')['sales_count'].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    # 删除有空值的行
    df = df.dropna()
    return df


def train():
    from xgboost import XGBRegressor

    print("从数据库加载数据...")
    df = load_data()
    print(f"数据量: {len(df)} 行")

    print("构建特征...")
    df = build_features(df)

    # 把品类名转成数字
    df['category_code'] = df['category'].astype('category').cat.codes
    category_mapping = dict(enumerate(df['category'].astype('category').cat.categories))

    features = ['month_num', 'year', 'lag_1', 'lag_2', 'rolling_mean_3', 'category_code']
    X = df[features]
    y = df['sales_count']

    print("训练 XGBoost 模型...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    # 训练 SHAP explainer
    # explainer = shap.TreeExplainer(model)

    # 保存模型
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, 'demand_forecast.pkl'), 'wb') as f:
        pickle.dump({
            'model': model,
            'category_mapping': category_mapping,
            'features': features
        }, f)

    print(f"模型已保存到 models/demand_forecast.pkl")
    print(f"训练完成，品类数量: {len(category_mapping)}")


if __name__ == '__main__':
    train()