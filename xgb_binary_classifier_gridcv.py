# code assumes required imports in place and train-test split completed

random_seed = 12
cv_k = 5

param_grid = {
    'max_depth': [3, 5, 6, 7, 8],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'learning_rate': [0.01, 0.025, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4]
}

xgb_clf = xgb.XGBClassifier(
    booster='gbtree',
    objective='binary:logistic',
    tree_method='hist',
    early_stopping_rounds=10,
    random_state=random_seed,
    eval_metric='logloss',
    verbosity=2,
    n_jobs=1,
    n_estimators=750
)

xgb_cv = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='accuracy',    # 'recall' / 'precision' / 'f1' / 'roc_auc'
    cv=cv_k,
    n_jobs=-1
)

xgb_cv.fit(X_train, y_train, eval_set=[(X_test, y_test)])
