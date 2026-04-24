import shap
import matplotlib.pyplot as plt


def get_shap_values(model, input_df):
    """Compute SHAP values for a single input row."""
    explainer = shap.Explainer(model)
    return explainer(input_df)


def plot_waterfall(shap_values, model_choice):
    """Waterfall plot for a single prediction. Returns (figure, raw shap vals array)."""
    if model_choice == "Random Forest":
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
        vals = shap_values.values[0][:, 1]
    else:
        shap.plots.waterfall(shap_values[0], show=False)
        vals = shap_values.values[0]
    return plt.gcf(), vals


def plot_global_summary(model, X_sample, model_choice):
    """Global SHAP bar summary across a background sample of rows."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    plt.figure(figsize=(10, 6))
    if model_choice == "Random Forest":
        shap.summary_plot(shap_values.values[:, :, 1], X_sample, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    return plt.gcf()


def plot_dependence(model, X_sample, feature, model_choice):
    """SHAP dependence plot for a specific feature."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    fig, ax = plt.subplots(figsize=(8, 5))
    if model_choice == "Random Forest":
        shap.dependence_plot(feature, shap_values.values[:, :, 1], X_sample, ax=ax, show=False)
    else:
        shap.dependence_plot(feature, shap_values.values, X_sample, ax=ax, show=False)
    return fig
