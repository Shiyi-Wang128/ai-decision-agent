import pickle
import os
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'demand_forecast.pkl')

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def predict_demand(category: str, month: int, year: int, lag_1: float, lag_2: float, rolling_mean_3: float):
    bundle = load_model()
    model = bundle['model']
    category_mapping = bundle['category_mapping']

    reverse_mapping = {v: k for k, v in category_mapping.items()}
    if category not in reverse_mapping:
        return None, f"品类 '{category}' 不在训练数据中"

    category_code = reverse_mapping[category]

    features = pd.DataFrame([{
        'month_num': month,
        'year': year,
        'lag_1': lag_1,
        'lag_2': lag_2,
        'rolling_mean_3': rolling_mean_3,
        'category_code': category_code
    }])

    prediction = model.predict(features)[0]

    # 用feature importance代替SHAP
    importance = model.feature_importances_
    feature_names = bundle['features']
    feature_impact = {feature_names[i]: round(float(importance[i]), 3)
                     for i in range(len(feature_names))}

    return round(float(prediction)), feature_impact