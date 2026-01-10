import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def squared_relative_error(y_true, y_pred):
    """
    计算平方和相对误差: sum( ((y_true - y_pred) / y_true)^2 )
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    epsilon = 1e-10
    y_true_safe = np.where(y_true == 0, epsilon, y_true)
    
    sre = ((y_true - y_pred) / y_true_safe) ** 2
    return np.sum(sre)

custom_scorer = make_scorer(squared_relative_error, greater_is_better=False)

def load_data(filename):
    df = pd.read_csv(filename, header=None, encoding='gbk')
    
    # 前30列(idx 0-29): 数值特征
    # 第31列(idx 30): 分类特征(药物)
    # 第32列(idx 31): 目标值
    X = df.iloc[:, :31] 
    y = df.iloc[:, 31]
    return X, y

def main():
    try:
        X_train, y_train = load_data('train.csv')
        X_test, y_test = load_data('test.csv')
    except FileNotFoundError:
        print("错误：未找到 train.csv 或 test.csv，请检查文件路径。")
        return

    numeric_features = list(range(30)) 
    categorical_features = [30]        

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # --- 构建 AdaBoost 模型 ---
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', AdaBoostRegressor(random_state=42)) 
    ])

    # --- 设置交叉验证超参数 ---
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 1.0],
        'regressor__loss': ['linear', 'square'] 
    }

    print("开始交叉验证训练...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=custom_scorer, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳超参数: {grid_search.best_params_}")

    # --- 测试集评估 ---
    y_pred = best_model.predict(X_test)

    relative_sample_errors = ((y_test - y_pred) / y_test) ** 2
    sample_errors = abs(y_test - y_pred)

    total_sre = np.sum(relative_sample_errors)
    mean_sre = np.mean(sample_errors)
    var_sre = np.var(sample_errors)

    print("=" * 40)
    print("AdaBoost 模型测试报告")
    print("=" * 40)
    print(f"1. 相对误差平方和 : {total_sre:.6f}")
    print(f"2. 样本误差均值   : {mean_sre:.6f}")
    print(f"3. 样本误差方差   : {var_sre:.6f}")
    print("=" * 40)

if __name__ == "__main__":
    main()