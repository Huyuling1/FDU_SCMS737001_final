import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

def squared_relative_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    epsilon = 1e-10
    y_true_safe = np.where(y_true == 0, epsilon, y_true)
    sre = ((y_true - y_pred) / y_true_safe) ** 2
    return np.sum(sre)

custom_scorer = make_scorer(squared_relative_error, greater_is_better=False)

def load_data(filename):
    df = pd.read_csv(filename, header=None, encoding='gbk')
    X = df.iloc[:, :31]
    y = df.iloc[:, 31]
    return X, y

def main():
    X_train, y_train = load_data('train.csv')
    X_test, y_test = load_data('test.csv')

    numeric_features = list(range(30))
    categorical_features = [30]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # 方案1: 更深的基学习器
    pipeline_adaboost_deep = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=10), random_state=42))
    ])

    param_grid_adaboost = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__learning_rate': [0.01, 0.05, 0.1, 0.5],
        'regressor__loss': ['linear', 'square', 'exponential']
    }

    # 方案2: 梯度提升树 (GBDT)
    pipeline_gbdt = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])

    param_grid_gbdt = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 5, 7],
        'regressor__subsample': [0.8, 0.9, 1.0]
    }

    # 方案3: XGBoost
    pipeline_xgb = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(random_state=42, objective='reg:squarederror'))
    ])

    param_grid_xgb = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__subsample': [0.8, 0.9, 1.0],
        'regressor__colsample_bytree': [0.8, 0.9, 1.0]
    }

    models = {
        'AdaBoost': (pipeline_adaboost_deep, param_grid_adaboost),
        'GBDT': (pipeline_gbdt, param_grid_gbdt),
        'XGBoost': (pipeline_xgb, param_grid_xgb)
    }

    results = {}
    
    for model_name, (pipeline, param_grid) in models.items():
        print(f"\n正在训练 {model_name}...")
        
        if param_grid:
            if len(param_grid) > 12:
                search = RandomizedSearchCV(
                    pipeline, param_grid, cv=5, scoring=custom_scorer,
                    n_iter=20, random_state=42, n_jobs=-1
                )
            else:
                search = GridSearchCV(
                    pipeline, param_grid, cv=5, scoring=custom_scorer,
                    n_jobs=-1
                )
        else:
            search = pipeline

        if param_grid:
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            print(f"最佳参数: {search.best_params_ if hasattr(search, 'best_params_') else 'N/A'}")
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline

        y_pred = best_model.predict(X_test)
        
        relative_sample_errors = ((y_test - y_pred) / y_test) ** 2
        sample_errors = abs(y_test - y_pred)
        
        total_sre = np.sum(relative_sample_errors)
        mean_sre = np.mean(sample_errors)
        var_sre = np.var(sample_errors)
        
        results[model_name] = {
            'total_sre': total_sre,
            'mean_sre': mean_sre,
            'var_sre': var_sre,
            'predictions': y_pred
        }
        
        print(f"{model_name} 结果:")
        print(f"  相对误差平方和: {total_sre:.6f}")
        print(f"  样本误差均值: {mean_sre:.6f}")
        print(f"  样本误差方差: {var_sre:.6f}")

    print("\n" + "="*50)
    print("模型性能对比")
    print("="*50)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15} | SRE: {metrics['total_sre']:.6f} | Mean: {metrics['mean_sre']:.6f} | Var: {metrics['var_sre']:.6f}")

    best_model_name = min(results, key=lambda x: results[x]['total_sre'])
    print(f"\n最佳模型: {best_model_name}")

if __name__ == "__main__":
    main()