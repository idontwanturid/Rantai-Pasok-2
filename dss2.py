import streamlit as st
import joblib
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable
import pandas as pd
from sklearn.linear_model import LinearRegression

# Mapping satuan
unit_mapping = {
    "Costs": "Rp",
    "Shipping costs": "Rp",
    "Price": "Rp",
    "Revenue generated": "Rp",
    "Order quantities": "unit",
    "Number of products sold":"Unit",
    "Manufacturing costs": "Rp",
    "Availability":"Unit",
    "Stock levels":"Unit",
    "Lead times":"Jam",
    "Shipping times":"Kali",
    "Manufacturing lead time":"Jam",
    "Production volumes":"Unit",
    "Defect rates":"%",
}

# Judul aplikasi
st.title("SCM Optima")

# Upload file dataset
uploaded_file = st.file_uploader("Unggah Dataset Anda (format CSV)", type=["csv"])

if uploaded_file is not None:
    # Membaca dataset
    df = pd.read_csv(uploaded_file)

    # Menampilkan dataset
    st.write("Dataset:")
    st.write(df.head())

    # Memfilter kolom numerik
    numeric_columns = df.select_dtypes(include=["float", "int"]).columns.tolist()

    if len(numeric_columns) < 2:
        st.error("Dataset harus memiliki minimal dua kolom numerik untuk fitur prediktif.")
    else:
        # Dropdown untuk memilih kolom
        st.write("Pilih kolom untuk fitur dan target:")
        factor1_col = st.selectbox("Variabel Bebas 1", options=numeric_columns)
        factor2_col = st.selectbox("Variabel Bebas 2", options=numeric_columns)
        demand_col = st.selectbox("Target Prediksi 1", options=numeric_columns)
        cost_col = st.selectbox("Target Prediksi 2", options=numeric_columns)

        # Ambil satuan berdasarkan mapping
        factor1_unit = unit_mapping.get(factor1_col, "")
        factor2_unit = unit_mapping.get(factor2_col, "")
        demand_unit = unit_mapping.get(demand_col, "")
        cost_unit = unit_mapping.get(cost_col, "")

        # Fitur dan target berdasarkan pilihan pengguna
        X = df[[factor1_col, factor2_col]]
        y_demand = df[demand_col]
        y_cost = df[cost_col]

        # Melatih model
        model_demand = LinearRegression().fit(X, y_demand)
        model_cost = LinearRegression().fit(X, y_cost)

        # Simpan model
        joblib.dump(model_demand, "model_demand.pkl")
        joblib.dump(model_cost, "model_cost.pkl")
        st.success("Models saved successfully!")

        # Load model prediktif
        model_demand = joblib.load("model_demand.pkl")
        model_cost = joblib.load("model_cost.pkl")

        # Input untuk prediksi (factor1 dan factor2)
        factor1 = st.number_input(
            f"Rata-rata {factor1_col} ({factor1_unit})",
            min_value=float(df[factor1_col].min()),
            value=float(df[factor1_col].mean()),
        )
        factor2 = st.number_input(
            f"Rata-rata {factor2_col} ({factor2_unit})",
            min_value=float(df[factor2_col].min()),
            value=float(df[factor2_col].mean()),
        )

        # Prediksi permintaan dan biaya per unit
        predicted_demand = model_demand.predict(np.array([[factor1, factor2]]))[0]
        predicted_cost = model_cost.predict(np.array([[factor1, factor2]]))[0]

        # Tampilkan hasil prediksi dengan satuan
        st.write(f"Prediksi 1: {predicted_demand:.2f} {demand_unit}")
        st.write(f"Prediksi 2: {predicted_cost:.2f} {cost_unit}")

        # Input kapasitas pengiriman maksimum
        max_capacity = st.number_input(
            f"Masukkan Kapasitas Pengiriman Maksimum (Unit)", min_value=0.0, value=0.0
        )

        # Mendefinisikan masalah Linear Programming
        model = LpProblem(name="Optimasi_Biaya_Pengiriman", sense=LpMaximize)

        # Variabel keputusan
        x = LpVariable("produk_dikirim", lowBound=0, cat="Continuous")

        # Fungsi objektif
        model += predicted_cost * x  # total biaya pengiriman

        # Kendala
        model += x <= predicted_demand  # Tidak melebihi permintaan
        model += x <= max_capacity  # Tidak melebihi kapasitas maksimum

        # Menyelesaikan model
        model.solve()

        # Menampilkan hasil
        if model.status == 1:  # Jika solusi optimal ditemukan
            produk_dikirim = x.value()
            total_biaya = predicted_cost * produk_dikirim
            st.write(f"Jumlah Produk yang Dikirim: {produk_dikirim:.2f} Unit")
            st.write(f"Total Biaya Pengiriman: {cost_unit}{total_biaya:.2f}")
        else:
            st.write("Tidak ada solusi optimal.")
else:
    st.info("Silakan unggah dataset untuk memulai.")
