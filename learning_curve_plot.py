# assuming all required imports in place

xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

results = xgb_eval.evals_result()

plt.figure(figsize=(8, 6))
sns.lineplot(data=results['validation_0']['rmse'], label='train')
sns.lineplot(data=results['validation_1']['rmse'], label='test')
plt.title("XGBoost - Learning Curves")
plt.xlabel('Number of Boosting Rounds')
plt.ylabel('RMSE')
plt.axvline(x=1000, color='firebrick', linestyle=':', label='chosen value')
plt.legend()
plt.show()
