import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Classificador de Câncer de Mama", layout="centered")
st.title("Classificador de Câncer de Mama")
st.write("Preveja se um tumor é benigno ou maligno baseado nas características fornecidas.")

model_path = "models/xgboost_breast_cancer_fs_optimized.pkl"
model = joblib.load(model_path)

features_info = {
    'texture_mean': (9.71 * 0.95, 39.28 * 1.05),
    'area_mean': (170.4 * 0.95, 2501.0 * 1.05),
    'smoothness_mean': (0.05263 * 0.95, 0.1634 * 1.05),
    'concavity_mean': (0.0, 0.4268 * 1.05),
    'symmetry_mean': (0.1167 * 0.95, 0.304 * 1.05),
    'area_se': (6.802 * 0.95, 542.2 * 1.05),
    'smoothness_se': (0.001713 * 0.95, 0.03113 * 1.05),
    'concavity_se': (0.0, 0.396 * 1.05),
    'symmetry_se': (0.007882 * 0.95, 0.06146 * 1.05),
    'fractal_dimension_se': (0.000895 * 0.95, 0.02984 * 1.05),
    'smoothness_worst': (0.07117 * 0.95, 0.2184 * 1.05),
    'symmetry_worst': (0.1565 * 0.95, 0.6638 * 1.05),
    'fractal_dimension_worst': (0.05504 * 0.95, 0.173 * 1.05)
}

st.subheader("Inserir valores das features")
input_data = {}
for feature, (min_val, max_val) in features_info.items():
    input_data[feature] = st.number_input(
        feature, min_value=float(min_val), max_value=float(max_val),
        value=float((min_val+max_val)/2), format="%.5f"
    )

if st.button("Prever"):
    input_df = pd.DataFrame([input_data])
    try:
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)

        with st.expander("Resultado da Predição", expanded=True):
            if prediction[0] == 0:
                st.success(f"Benigno(Probabilidade: {proba[0][0]:.2f})")
            else:
                st.error(f"Maligno(Probabilidade: {proba[0][1]:.2f})")

        with st.expander("Nota sobre interpretabilidade", expanded=True):
            st.info("Para análise detalhada das features e impacto em cada predição, consulte o notebook com SHAP.")

    except Exception as e:
        st.error(f"Erro ao fazer a previsão: {e}")
