import pathlib
import yaml
import pandas as pd
import xgboost as xgb
import optuna
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_slice, plot_contour, plot_param_importances
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def load_data(file_path):
    return pd.read_csv(file_path)


def try_models(X,y,test_path):
    for name, model in [
            # ("xgboost", xgb.XGBRegressor()),
            # ("random_forest", RandomForestRegressor()),
            # ("gradient_boosting", GradientBoostingRegressor()),
            ('lgm',LGBMRegressor(boosting_type='goss',device='gpu')),
            ('cbr',CatBoostRegressor())
        ]:
            with mlflow.start_run():
                mlflow.log_param('model',name)
                
                test=pd.read_csv(test_path)
                test_X = test.drop(columns="target")
                test_y = test["target"]
                
                model_=model
                
                model_.fit(X,y)
                pred_y=model_.predict(test_X)
                
                mlflow.log_metric('r2_score',r2_score(test_y,pred_y))
                mlflow.log_metric('mse',mean_squared_error(test_y,pred_y))
                mlflow.log_metric('mae',mean_absolute_error(test_y,pred_y))


def objective1(trial):
    global X,y
    n_estimators_ = trial.suggest_int('n_estimators', 10, 200)
    max_depth_ = trial.suggest_int('max_depth', 3, 25)
    learning_rate_ = trial.suggest_float('learning_rate', 0.00001, 1,log=True)
    crietion_=trial.suggest_categorical('criterion',["squared_error",'friedman_mse'])
    loss_=trial.suggest_categorical('loss',["squared_error",'absolute_error','huber','quantile'])

    model = GradientBoostingRegressor(
        n_estimators=n_estimators_,
        max_depth=max_depth_,
        learning_rate=learning_rate_,
        criterion=crietion_,
        loss=loss_,
        random_state=42
    )

    score = cross_val_score(model, X=X, y=y, cv=5,n_jobs=-1).mean()

    return score

def objective2(trial):
    global X,y
    n_estimators_ = trial.suggest_int('n_estimators', 10, 200)
    max_depth_ = trial.suggest_int('max_depth', 3, 25)
    max_features_ = trial.suggest_float('max_features', 0, 1)
    max_samples_ = trial.suggest_float('max_samples', 0, 1)
    bootstrap_=trial.suggest_categorical('bootstrap',[True,False])
    crietion_=trial.suggest_categorical('criterion',["squared_error","absolute_error",'friedman_mse'])
    model=None

    if  bootstrap_:
        model = RandomForestRegressor(
            n_estimators=n_estimators_,
            max_depth=max_depth_,
            max_samples=max_samples_,
            max_features=max_features_,
            bootstrap=bootstrap_,
            criterion=crietion_,
            random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators_,
            max_depth=max_depth_,
            max_features=max_features_,
            bootstrap=bootstrap_,
            criterion=crietion_,
            random_state=42
        )
        
    score = cross_val_score(model, X=X, y=y, cv=5,n_jobs=-1).mean()

    return score 

def objective3(trial):
    global X,y
    n_estimators_ = trial.suggest_int('n_estimators', 10, 200)
    max_depth_ = trial.suggest_int('max_depth', 3, 25)
    eta_ = trial.suggest_float('eta', 0.01, 1,log=True)
    booster_=trial.suggest_categorical('booster',["gbtree",'dart','gblinear'])
    sampling_method_=trial.suggest_categorical('sampling_method',["uniform",'gradient_based'])
    tree_method_=trial.suggest_categorical('tree_method',["exact",'hist','approx'])
    process_type_=trial.suggest_categorical('process_type',["default",'update'])

    model = xgb.XGBRegressor(
        n_estimators=n_estimators_,
        max_depth=max_depth_,
        learning_rate=eta_,
        booster=booster_,
        sampling_method=sampling_method_,
        tree_method=tree_method_,
        process_type=process_type_,n_jobs=-1
    )

    score = cross_val_score(model, X=X, y=y, cv=5,n_jobs=-1).mean()

    return score


def objective4(trial):
    global X,y
    n_estimators_ = trial.suggest_int('n_estimators', 10, 250)
    max_depth_ = trial.suggest_int('max_depth', 3, 50)
    subsample_ = trial.suggest_float('subsample', 0, 1)
    top_rate_ = trial.suggest_float('top_rate', 0, 1)
    other_rate_ = trial.suggest_float('other_rate', 0, 0.999999-top_rate_)
    reg_alpha_ = trial.suggest_float('reg_alpha', 0, 10)
    reg_lambda_ = trial.suggest_float('reg_lambda', 0, 10)
    colsample_bytree_ = trial.suggest_float('colsample_bytree', 0, 1)
    colsample_bynode_ = trial.suggest_float('colsample_bynode', 0, 1)
    learning_rate_ = trial.suggest_float('learning_rate', 0.01, 1,log=True)
    boosting_type_=trial.suggest_categorical('boosting_type',["gbdt",'dart','rf'])
    data_sample_strategy_=trial.suggest_categorical('data_sample_strategy',["goss",'bagging'])
    device_type_=trial.suggest_categorical('device_type',["gpu"])
    metric_=trial.suggest_categorical('metric',["mae"])

    model = LGBMRegressor(
        n_estimators=n_estimators_,
        max_depth=max_depth_,
        learning_rate=learning_rate_,
        subsample=subsample_,
        top_rate=top_rate_,
        other_rate=other_rate_,
        reg_alpha=reg_alpha_,
        reg_lambda=reg_lambda_,
        colsample_bytree=colsample_bytree_,
        colsample_bynode_=colsample_bynode_,
        boosting_type=boosting_type_,
        data_sample_strategy=data_sample_strategy_,
        device_type=device_type_,
        metric=metric_
    )

    score = cross_val_score(model, X=X, y=y, cv=5,n_jobs=-1).mean()

    return score

def tune_lgbm(X,y,test_path):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective4, n_trials=200,n_jobs=-1) 
    
    best_trial = study.best_trial
    
    with mlflow.start_run():
        mlflow.log_param('model','lighboost_tunned')

        for key,value in best_trial.params.items():
            mlflow.log_param(key,value)

        
        test=pd.read_csv(test_path)
        test_X = test.drop(columns="target")
        test_y = test["target"]
        
        model_=LGBMRegressor(**best_trial.params,random_state=42)
        
        model_.fit(X,y)
        pred_y=model_.predict(test_X)
        
        mlflow.log_metric('r2_score',r2_score(test_y,pred_y))
        mlflow.log_metric('mse',mean_squared_error(test_y,pred_y))
        mlflow.log_metric('mae',mean_absolute_error(test_y,pred_y))
        fig=plot_optimization_history(study)
        mlflow.log_figure(fig,'optuna_optimization_history.png')
        fig=plot_param_importances(study)
        mlflow.log_figure(fig,'optuna_param_importance.png')
        fig=plot_slice(study)
        mlflow.log_figure(fig,'optuna_plot_slice.png')
        
def tune_xgboost(X,y,test_path):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective3, n_trials=150,n_jobs=-1) 
    
    best_trial = study.best_trial
    
    with mlflow.start_run():
        mlflow.log_param('model','XGboost_tunned')

        for key,value in best_trial.params.items():
            mlflow.log_param(key,value)

        
        test=pd.read_csv(test_path)
        test_X = test.drop(columns="target")
        test_y = test["target"]
        
        model_=xgb.XGBRegressor(**best_trial.params,random_state=42)
        
        model_.fit(X,y)
        pred_y=model_.predict(test_X)
        
        mlflow.log_metric('r2_score/metric',r2_score(test_y,pred_y))
        mlflow.log_metric('mse/metric',mean_squared_error(test_y,pred_y))
        mlflow.log_metric('mae/metric',mean_absolute_error(test_y,pred_y))
        fig=plot_optimization_history(study)
        mlflow.log_figure(fig,'optuna_optimization_history.png')
        fig=plot_param_importances(study)
        mlflow.log_figure(fig,'optuna_param_importance.png')
        fig=plot_slice(study)
        mlflow.log_figure(fig,'optuna_plot_slice.png')
        
def tune_random_forest(X,y,test_path):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective2, n_trials=150,n_jobs=-1) 
    
    best_trial = study.best_trial
    
    with mlflow.start_run():
        mlflow.log_param('model','random_forest_tunned')

        for key,value in best_trial.params.items():
            mlflow.log_param(key,value)

        
        test=pd.read_csv(test_path)
        test_X = test.drop(columns="target")
        test_y = test["target"]
        
        model_=RandomForestRegressor(**best_trial.params,random_state=42)
        
        model_.fit(X,y)
        pred_y=model_.predict(test_X)
        
        mlflow.log_metric('r2_score/metric',r2_score(test_y,pred_y))
        mlflow.log_metric('mse/metric',mean_squared_error(test_y,pred_y))
        mlflow.log_metric('mae/metric',mean_absolute_error(test_y,pred_y))
        fig=plot_optimization_history(study)
        mlflow.log_figure(fig,'optuna_optimization_history.png')
        fig=plot_param_importances(study)
        mlflow.log_figure(fig,'optuna_param_importance.png')
        fig=plot_slice(study)
        mlflow.log_figure(fig,'optuna_plot_slice.png')

def tune_gradient_boosting(X,y,test_path):
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective1, n_trials=150,n_jobs=-1) 
    
    best_trial = study.best_trial
    
    with mlflow.start_run():
        mlflow.log_param('model','gradient_boosting_tunned')

        for key,value in best_trial.params.items():
            mlflow.log_param(key,value)

        
        test=pd.read_csv(test_path)
        test_X = test.drop(columns="target")
        test_y = test["target"]
        
        model_=GradientBoostingRegressor(**best_trial.params,random_state=42)
        
        model_.fit(X,y)
        pred_y=model_.predict(test_X)
        
        mlflow.log_metric('r2_score/metric',r2_score(test_y,pred_y))
        mlflow.log_metric('mse/metric',mean_squared_error(test_y,pred_y))
        mlflow.log_metric('mae/metric',mean_absolute_error(test_y,pred_y))
        fig=plot_optimization_history(study)
        mlflow.log_figure(fig,'optuna_optimization_history.png')
        fig=plot_param_importances(study)
        mlflow.log_figure(fig,'optuna_param_importance.png')
        fig=plot_slice(study)
        mlflow.log_figure(fig,'optuna_plot_slice.png')    

X,y=None,None
def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent
    data_path = home_dir / "data" / "train_test_split" / "train.csv"
    test_path = home_dir / "data" / "train_test_split" / "test.csv"
    params = None
    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["model"]

    df = load_data(data_path)
    global X,y
    X = df.drop(columns="target")
    y = df["target"]

    if params["try_model"]:
        try_models(X,y,test_path)
        
    if params['tune_gradient_boosting']:
        tune_gradient_boosting(X,y,test_path)
        
    if params['tune_random_forest']:
        tune_random_forest(X,y,test_path)
        
    if params['tune_XGBoost']:
        tune_xgboost(X,y,test_path)
        
    if params['tune_light_boost']:
        tune_lgbm(X,y,test_path)
    return


if __name__ == "__main__":
    main()
