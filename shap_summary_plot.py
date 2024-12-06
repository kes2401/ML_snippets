# assuming all required imports completed, including shap

explainer = shap.Explainer(best_xgb_model)

shap_values = explainer(X_train)

shap.summary_plot(shap_values_xgb, X_train)
