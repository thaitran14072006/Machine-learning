"""
preprocess.py - Tiền xử lý dữ liệu và các hàm dùng chung
Bộ dữ liệu: House Price Prediction Dataset (biến mục tiêu: Price)
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

DATA_PATH = "Data/House Price Prediction Dataset.csv"
TARGET = "Price"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ----------------------------------------------------------------------------
# 1. Đọc dữ liệu và làm sạch
# ----------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    """Đọc file CSV."""
    return pd.read_csv(path)


def drop_id(df):
    """Bỏ cột Id (chỉ là số thứ tự, không mang thông tin dự đoán)."""
    return df.drop(columns=["Id"], errors="ignore")


def check_duplicates(df, remove=True):
    """Đếm dòng trùng lặp; mặc định loại bỏ."""
    n_dup = int(df.duplicated().sum())
    if remove and n_dup > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    return df, n_dup


def detect_outliers_iqr(df, columns=None, factor=1.5):
    """Thống kê outlier theo quy tắc IQR cho từng cột số."""
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - factor * iqr, q3 + factor * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({"Cột": col, "Q1": q1, "Q3": q3, "Cận dưới": low,
                     "Cận trên": high, "Số outlier": n_out,
                     "Tỉ lệ (%)": 100 * n_out / len(df)})
    return pd.DataFrame(rows)


def remove_outliers_iqr(df, columns, factor=1.5):
    """Loại các dòng có outlier (không dùng mặc định, chỉ để tham khảo)."""
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask &= df[col].between(q1 - factor * iqr, q3 + factor * iqr)
    return df[mask].reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. One-hot encoding, chia train/test, chuẩn hóa
# ----------------------------------------------------------------------------
def encode_onehot(df):
    """One-hot encoding cho các cột phân loại (drop_first tránh đa cộng tuyến)."""
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    return pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)


def preprocess(path=DATA_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE,
               remove_outliers=False):
    """
    Quy trình đầy đủ: đọc -> bỏ Id -> bỏ trùng lặp -> (tuỳ chọn) bỏ outlier
    -> one-hot -> chia train/test 80/20 -> chuẩn hóa (fit trên train).
    Trả về: X_train, X_test, y_train, y_test, scaler
    """
    df = drop_id(load_data(path))
    df, _ = check_duplicates(df, remove=True)
    if remove_outliers:
        df = remove_outliers_iqr(df, df.select_dtypes(include=np.number).columns)
    df = encode_onehot(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train),
                           columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test),
                          columns=X.columns, index=X_test.index)
    return X_train, X_test, y_train, y_test, scaler


# Dữ liệu dùng chung: nạp một lần khi import file này
X_train, X_test, y_train, y_test, scaler = preprocess()
FEATURES = X_train.columns.tolist()


# ----------------------------------------------------------------------------
# 3. Mô hình và hàm đánh giá dùng chung
# ----------------------------------------------------------------------------
def get_models():
    """Ba mô hình dùng chung cho toàn nhóm. SVR được chuẩn hóa thêm biến mục tiêu
    vì Price có đơn vị lớn (cỡ 10^5 - 10^6)."""
    return {
        "SVR": TransformedTargetRegressor(
            regressor=SVR(kernel="rbf", C=1.0, epsilon=0.1),
            transformer=StandardScaler()),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }


def evaluate(model, features):
    """
    Huấn luyện `model` trên tập train với danh sách `features`, dự đoán trên test.
    Trả về dict: R2, RMSE, MAE, Time (giây; gồm huấn luyện + dự đoán).
    """
    features = list(features)
    est = clone(model)
    start = time.perf_counter()
    est.fit(X_train[features], y_train)
    y_pred = est.predict(X_test[features])
    elapsed = time.perf_counter() - start
    return {
        "R2": r2_score(y_test, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAE": mean_absolute_error(y_test, y_pred),
        "Time": elapsed,
    }


def run_experiment(method, features):
    """Chạy cả 3 mô hình với một tập đặc trưng, trả về DataFrame kết quả."""
    features = list(features)
    rows = []
    for name, model in get_models().items():
        res = evaluate(model, features)
        rows.append({"Method": method, "Model": name,
                     "n_features": len(features),
                     "Features": ", ".join(features), **res})
    return pd.DataFrame(rows)
