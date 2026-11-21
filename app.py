import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Angka Harapan Hidup", page_icon="🌍", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model_life_expectancy.pkl")

meta = load_model()          # dict: pipeline, features, defaults, min, max, metrik
pipeline = meta["pipeline"]  # StandardScaler + Gradient Boosting
FEATURES = meta["features"]

LABELS = {
    "Status_enc": "Status negara",
    "Adult_mortality": "Adult Mortality (kematian dewasa per 1000 penduduk)",
    "Infant_deaths": "Infant Deaths (kematian bayi per 1000 penduduk)",
    "Alcohol": "Konsumsi alkohol (liter per kapita)",
    "Percentage_expenditure": "Pengeluaran kesehatan (% dari GDP per kapita)",
    "Hepatitis_B": "Imunisasi Hepatitis B (% anak usia 1 tahun)",
    "Measles": "Kasus campak (per 1000 penduduk)",
    "BMI": "Rata-rata BMI",
    "Polio": "Imunisasi Polio (% anak usia 1 tahun)",
    "Total_expenditure": "Pengeluaran kesehatan pemerintah (% total anggaran)",
    "Diphtheria": "Imunisasi Difteri/DTP3 (% anak usia 1 tahun)",
    "HIV_AIDS": "Kematian HIV/AIDS (per 1000 kelahiran, usia 0-4 th)",
    "GDP": "GDP per kapita (USD)",
    "Population": "Jumlah populasi",
    "Thinness_1_19": "Prevalensi kurus usia 10-19 tahun (%)",
    "Income_composition": "Income Composition of Resources (0-1)",
    "Schooling": "Rata-rata lama sekolah (tahun)",
}
GROUPS = {
    "Kesehatan": ["Adult_mortality", "Infant_deaths", "HIV_AIDS", "Measles", "BMI", "Thinness_1_19", "Alcohol"],
    "Imunisasi": ["Hepatitis_B", "Polio", "Diphtheria"],
    "Ekonomi": ["GDP", "Percentage_expenditure", "Total_expenditure", "Population", "Income_composition"],
    "Pendidikan & Status": ["Schooling", "Status_enc"],
}

st.title("🌍 Prediksi Angka Harapan Hidup (Life Expectancy)")
st.caption(
    f"Model: {meta['model_name']} | R² data validasi: {meta['r2']:.3f} | MAE: {meta['mae']:.2f} tahun. "
    "Data: WHO Life Expectancy (2000-2015)."
)

tab_pred, tab_info = st.tabs(["Prediksi", "Tentang Model"])

with tab_pred:
    st.write("Atur nilai indikator di bawah ini, lalu klik **Prediksi**.")
    values = {}
    cols = st.columns(len(GROUPS))
    for col, (group, feats) in zip(cols, GROUPS.items()):
        with col:
            st.subheader(group)
            for f in feats:
                if f not in FEATURES:
                    continue
                if f == "Status_enc":
                    s = st.selectbox(LABELS[f], ["Developing", "Developed"])
                    values[f] = 1 if s == "Developed" else 0
                    continue
                lo, hi, dflt = meta["min"][f], meta["max"][f], meta["defaults"][f]
                if f == "Population":
                    values[f] = st.number_input(LABELS[f], min_value=float(lo), max_value=float(hi), value=float(dflt), step=100000.0)
                else:
                    values[f] = st.slider(LABELS[f], float(lo), float(hi), float(dflt))

    if st.button("Prediksi", type="primary"):
        X = pd.DataFrame([values])[FEATURES]
        pred = float(pipeline.predict(X)[0])
        st.metric("Prediksi Angka Harapan Hidup", f"{pred:.1f} tahun")
        st.caption(f"Perkiraan rentang ±{meta['mae']:.1f} tahun (berdasarkan MAE data validasi).")
        if pred >= 75:
            st.success("Kategori tinggi (≥ 75 tahun)")
        elif pred >= 60:
            st.info("Kategori menengah (60-75 tahun)")
        else:
            st.warning("Kategori rendah (< 60 tahun)")

with tab_info:
    st.subheader("Variabel paling berpengaruh")
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=[LABELS.get(f, f) for f in FEATURES]).sort_values(ascending=False)
        st.bar_chart(imp.head(10))
    st.markdown(
        "- Model dilatih pada data yang sudah dibersihkan (missing value, outlier, dan skala populasi dikoreksi).\n"
        "- Prediksi bersifat statistik dan **bukan** pengganti analisis kesehatan resmi."
    )
