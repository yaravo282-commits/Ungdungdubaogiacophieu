import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# FIX SEED (ổn định kết quả)
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Giao diện Streamlit
st.title("📈 Mô hình Dự báo Giá Cổ phiếu bằng LSTM")

# Nhập mã cổ phiếu và chọn ngày
ticker = st.text_input("Mã cổ phiếu (theo yfinance, ví dụ: VNM.VN, VIC, AAPL):", "VNM.VN")
start_date = st.date_input("Ngày bắt đầu:", pd.to_datetime("2018-01-01"))
end_date = st.date_input("Ngày kết thúc:", pd.to_datetime("2026-10-14"))

if st.button("Chạy dự báo"):
    with st.spinner("Đang tải dữ liệu và huấn luyện mô hình..."):
        # Tải dữ liệu
        df = yf.download(ticker, start=start_date, end=end_date)

        if df.empty:
            st.error("Không tìm thấy dữ liệu cổ phiếu! Vui lòng kiểm tra lại mã.")
        else:
            data = df[['Close']].values

            train_size = int(0.8 * len(data))
            train_data = data[:train_size]
            test_data = data[train_size:]

            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_data)
            test_scaled = scaler.transform(test_data)

            def create_sequences(data, window_size=60):
                X, y = [], []
                for i in range(window_size, len(data)):
                    X.append(data[i - window_size : i])
                    y.append(data[i])
                return np.array(X), np.array(y)

            X_train, y_train = create_sequences(train_scaled)
            test_combined = np.vstack((train_scaled[-60:], test_scaled))
            X_test, y_test = create_sequences(test_combined)

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            val_size = int(0.1 * len(X_train))
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train_final = X_train[:-val_size]
            y_train_final = y_train[:-val_size]

            # Xây dựng mô hình LSTM
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            early_stop = EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )

            history = model.fit(
                X_train_final,
                y_train_final,
            validation_data=(X_val, y_val),
                epochs=10,
                batch_size=32,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0,
            )

            # Dự báo dữ liệu test
            test_pred = model.predict(X_test)
            test_pred = scaler.inverse_transform(test_pred)
            y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Tính chỉ số đánh giá
            rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
            mae = mean_absolute_error(y_test_real, test_pred)
            r2 = r2_score(y_test_real, test_pred)

            st.subheader("📊 Đánh giá mô hình")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{rmse:.2f}")
            col2.metric("MAE", f"{mae:.2f}")
            col3.metric("R2 Score", f"{r2:.4f}")

            # Dự báo 10 ngày tới
            last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
            last_60_days = last_60_days.reshape(1, 60, 1)

            future_predictions = []
            for i in range(10):
                next_pred = model.predict(last_60_days, verbose=0)
                future_predictions.append(next_pred[0, 0])
                last_60_days = np.append(
                    last_60_days[:, 1:, :],
                    next_pred.reshape(1, 1, 1),
                    axis=1,
                )

            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = scaler.inverse_transform(future_predictions)

            # Vẽ biểu đồ kết quả
            st.subheader("📈 Biểu đồ Giá thực tế vs Dự báo 10 ngày tới")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(y_test_real, label="Giá thật")
            ax.plot(test_pred, label="Dự báo test")

            future_x = range(len(y_test_real), len(y_test_real) + 10)
            ax.plot(
                future_x,
                future_predictions,
                marker='o',
                color='red',
                label="Dự báo 10 ngày tới",
            )
            ax.legend()
            st.pyplot(fig)

            # Hiển thị bảng kết quả 10 ngày
            st.subheader("📋 Bảng giá dự báo 10 ngày tới:")
            df_future = pd.DataFrame(
                future_predictions,
                columns=["Giá dự báo"],
                index=[f"Ngày +{i+1}" for i in range(10)],
            )
            st.dataframe(df_future)
