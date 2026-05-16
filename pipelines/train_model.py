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

    df['month_num'] = df['month'].dt.month
    df['year'] = df['month'].dt.year

    df['lag_1'] = df.groupby('category')['sales_count'].shift(1)
    df['lag_2'] = df.groupby('category')['sales_count'].shift(2)
    df['rolling_mean_3'] = df.groupby('category')['sales_count'].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    df = df.dropna()
    return df


def evaluate(model, X_test, y_test, df_test, category_mapping):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # MAPE: skip rows where actual = 0 to avoid division by zero
    mask = y_test > 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

    print("\n" + "="*50)
    print("模型评估结果（Holdout Test Set）")
    print("="*50)
    print(f"  MAE  (平均绝对误差):     {mae:.1f} 件")
    print(f"  RMSE (均方根误差):        {rmse:.1f} 件")
    print(f"  MAPE (平均绝对百分比误差): {mape:.1f}%")

    # Per-category MAE for top 10 categories by test volume
    df_test['category_name'] = df_test['category_code'].astype(int).map(category_mapping)
    df_test = df_test.copy()
    df_test['y_pred'] = y_pred
    df_test['abs_error'] = np.abs(y_test.values - y_pred)


    cat_mae = (
        df_test.groupby('category_name')
        .agg(mae=('abs_error', 'mean'), count=('abs_error', 'count'))
        .sort_values('count', ascending=False)
        .head(10)
    )
    print("\nTop 10 品类（按测试集样本量）的 MAE：")
    print(cat_mae.to_string())
    print("="*50 + "\n")

    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}


def train():
    from xgboost import XGBRegressor

    print("从数据库加载数据...")
    df = load_data()
    print(f"原始数据量: {len(df)} 行")

    print("构建特征...")
    df = build_features(df)
    print(f"特征构建后数据量: {len(df)} 行")

    df['category_code'] = df['category'].astype('category').cat.codes
    category_mapping = dict(enumerate(df['category'].astype('category').cat.categories))

    features = ['month_num', 'year', 'lag_1', 'lag_2', 'rolling_mean_3', 'category_code']
    X = df[features]
    y = df['sales_count']

    # 时间序列切分：最后3个月作为测试集（不能随机 shuffle，否则数据泄露）
    cutoff = df['month'].max() - pd.DateOffset(months=3)
    train_mask = df['month'] <= cutoff
    test_mask = df['month'] > cutoff

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"训练集: {len(X_train)} 行 | 测试集: {len(X_test)} 行")
    print(f"测试集时间范围: {df[test_mask]['month'].min().strftime('%Y-%m')} 到 {df[test_mask]['month'].max().strftime('%Y-%m')}")

    print("\n训练 XGBoost 模型...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # 评估
    metrics = evaluate(model, X_test, y_test, df[test_mask].copy(), category_mapping)

    # Feature importance
    importance = model.feature_importances_
    print("特征重要性：")
    for feat, imp in sorted(zip(features, importance), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    # 保存模型 + metrics
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, 'demand_forecast.pkl'), 'wb') as f:
        pickle.dump({
            'model': model,
            'category_mapping': category_mapping,
            'features': features,
            'eval_metrics': metrics,          # 存进去，README 和前端可以直接读
            'train_cutoff': str(cutoff.date()),
        }, f)

    print(f"模型已保存，品类数量: {len(category_mapping)}")
    print(f"评估指标已写入 pkl：{metrics}")


if __name__ == '__main__':
    train()