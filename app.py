import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Dự báo giá cổ phiếu bằng LSTM",
    page_icon="📈",
    layout="wide"
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

st.title("📈 DỰ BÁO GIÁ CỔ PHIẾU BẰNG LSTM")
st.write("Ứng dụng dự báo giá cổ phiếu bằng mô hình LSTM.")

st.warning(
    "Kết quả chỉ mang tính tham khảo, không phải lời khuyên đầu tư."
)

with st.sidebar:
    st.header("⚙️ Cài đặt")

    ticker = st.text_input(
        "Mã cổ phiếu",
        value="VNM.VN"
    ).strip().upper()

    start_date = st.date_input(
        "Ngày bắt đầu",
        value=pd.Timestamp("2018-01-01").date()
    )

    end_date = st.date_input(
        "Ngày kết thúc",
        value=pd.Timestamp.today().date()
    )

    window_size = st.slider(
        "Số ngày dữ liệu đầu vào",
        min_value=10,
        max_value=120,
        value=60
    )

    epochs = st.slider(
        "Số epoch huấn luyện",
        min_value=1,
        max_value=50,
        value=10
    )

    batch_size = st.selectbox(
        "Batch size",
        options=[16, 32, 64],
        index=1
    )

    forecast_days = st.slider(
        "Số ngày dự báo tương lai",
        min_value=1,
        max_value=30,
        value=10
    )

    run_button = st.button(
        "🚀 Chạy mô hình LSTM",
        use_container_width=True
    )


@st.cache_data(ttl=3600)
def download_data(ticker, start_date, end_date):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        return pd.DataFrame()

    df = df[["Close"]].copy()
    df.columns = ["Close"]

    return df.dropna()


def create_sequences(data, window_size=60):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def build_model(window_size):
    model = Sequential([
        LSTM(
            50,
            return_sequences=True,
            input_shape=(window_size, 1)
        ),
        LSTM(50),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


if run_button:

    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    if start_date >= end_date:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        st.stop()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    with st.spinner("Đang tải dữ liệu cổ phiếu..."):
        try:
            df = download_data(
                ticker,
                str(start_date),
                str(end_date + pd.Timedelta(days=1))
            )
        except Exception as e:
            st.error("Không thể tải dữ liệu.")
            st.code(str(e))
            st.stop()

    if df.empty:
        st.error(
            "Không tìm thấy dữ liệu. Hãy kiểm tra mã cổ phiếu "
            "hoặc thử lại sau."
        )
        st.stop()

    data = df[["Close"]].values

    if len(data) <= window_size + 30:
        st.error(
            "Không đủ dữ liệu. Hãy chọn khoảng thời gian dài hơn "
            "hoặc giảm số ngày dữ liệu đầu vào."
        )
        st.stop()

    st.header(f"📊 Dữ liệu cổ phiếu {ticker}")

    latest_price = float(df["Close"].iloc[-1])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Giá đóng cửa gần nhất",
        f"{latest_price:,.2f}"
    )

    col2.metric(
        "Số phiên giao dịch",
        f"{len(df):,}"
    )

    col3.metric(
        "Ngày cập nhật",
        df.index[-1].strftime("%d/%m/%Y")
    )

    st.dataframe(
        df.tail(20),
        use_container_width=True
    )

    fig_history, ax_history = plt.subplots(figsize=(12, 5))

    ax_history.plot(
        df.index,
        df["Close"],
        label="Giá đóng cửa"
    )

    ax_history.set_title(f"Lịch sử giá cổ phiếu {ticker}")
    ax_history.set_xlabel("Ngày")
    ax_history.set_ylabel("Giá")
    ax_history.legend()
    ax_history.grid(True, alpha=0.3)

    fig_history.tight_layout()
    st.pyplot(fig_history)
    plt.close(fig_history)

    train_size = int(0.8 * len(data))

    train_data = data[:train_size]
    test_data = data[train_size:]

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(
        train_scaled,
        window_size
    )

    test_combined = np.vstack([
        train_scaled[-window_size:],
        test_scaled
    ])

    X_test, y_test = create_sequences(
        test_combined,
        window_size
    )

    X_train = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )

    X_test = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1],
        1
    )

    val_size = max(1, int(0.1 * len(X_train)))

    if len(X_train) - val_size < 1:
        st.error("Không đủ dữ liệu để huấn luyện mô hình.")
        st.stop()

    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]

    st.header("🧠 Huấn luyện mô hình LSTM")

    progress = st.progress(0)
    status = st.empty()

    try:
        model = build_model(window_size)

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        status.info("Đang huấn luyện mô hình...")

        history = model.fit(
            X_train_final,
            y_train_final,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

        progress.progress(100)
        status.success("Huấn luyện thành công!")

    except Exception as e:
        st.error("Lỗi khi huấn luyện mô hình.")
        st.code(str(e))
        st.stop()

    st.header("📉 Đánh giá mô hình")

    test_pred_scaled = model.predict(
        X_test,
        verbose=0
    )

    test_pred = scaler.inverse_transform(
        test_pred_scaled
    )

    y_test_real = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )

    rmse = np.sqrt(
        mean_squared_error(y_test_real, test_pred)
    )

    mae = mean_absolute_error(
        y_test_real,
        test_pred
    )

    r2 = r2_score(
        y_test_real,
        test_pred
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("RMSE", f"{rmse:,.4f}")
    col2.metric("MAE", f"{mae:,.4f}")
    col3.metric("R²", f"{r2:.4f}")

    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))

    ax_loss.plot(
        history.history["loss"],
        label="Train Loss"
    )

    ax_loss.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    ax_loss.set_title("Loss Train vs Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Mean Squared Error")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    fig_loss.tight_layout()
    st.pyplot(fig_loss)
    plt.close(fig_loss)

    st.subheader("So sánh giá thực tế và giá dự báo")

    test_dates = df.index[train_size:]

    fig_test, ax_test = plt.subplots(figsize=(12, 6))

    ax_test.plot(
        test_dates,
        y_test_real.flatten(),
        label="Giá thật"
    )

    ax_test.plot(
        test_dates,
        test_pred.flatten(),
        label="Giá dự báo"
    )

    ax_test.set_title(f"Kết quả dự báo trên tập kiểm tra: {ticker}")
    ax_test.set_xlabel("Ngày")
    ax_test.set_ylabel("Giá")
    ax_test.legend()
    ax_test.grid(True, alpha=0.3)

    fig_test.tight_layout()
    st.pyplot(fig_test)
    plt.close(fig_test)

    st.header(f"🔮 Dự báo {forecast_days} ngày tiếp theo")

    all_scaled = np.vstack([
        train_scaled,
        test_scaled
    ])

    last_days = all_scaled[-window_size:].copy()
    future_predictions = []

    for _ in range(forecast_days):
        X_input = last_days.reshape(
            1,
            window_size,
            1
        )

        next_pred = model.predict(
            X_input,
            verbose=0
        )

        predicted_value = float(next_pred[0, 0])
        future_predictions.append(predicted_value)

        last_days = np.vstack([
            last_days[1:],
            [[predicted_value]]
        ])

    future_predictions = np.array(
        future_predictions
    ).reshape(-1, 1)

    future_predictions = scaler.inverse_transform(
        future_predictions
    ).flatten()

    future_dates = pd.bdate_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days
    )

    forecast_df = pd.DataFrame({
        "Ngày dự báo": future_dates.strftime("%d/%m/%Y"),
        "Giá dự báo": np.round(future_predictions, 2)
    })

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )

    st.metric(
        f"Giá dự báo sau {forecast_days} ngày",
        f"{future_predictions[-1]:,.2f}",
        f"{future_predictions[-1] - latest_price:+,.2f}"
    )

    fig_future, ax_future = plt.subplots(figsize=(12, 6))

    display_days = min(120, len(df))

    ax_future.plot(
        df.index[-display_days:],
        df["Close"].iloc[-display_days:],
        label="Giá lịch sử"
    )

    ax_future.plot(
        future_dates,
        future_predictions,
        marker="o",
        label="Giá dự báo"
    )

    ax_future.set_title(f"Dự báo giá cổ phiếu {ticker}")
    ax_future.set_xlabel("Ngày")
    ax_future.set_ylabel("Giá")
    ax_future.legend()
    ax_future.grid(True, alpha=0.3)

    fig_future.tight_layout()
    st.pyplot(fig_future)
    plt.close(fig_future)

    csv_data = forecast_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 Tải kết quả dự báo CSV",
        data=csv_data,
        file_name=f"du_bao_{ticker}.csv",
        mime="text/csv"
    )

    st.success("Đã hoàn thành dự báo!")

else:
    st.info(
        "Chọn mã cổ phiếu ở thanh bên trái, sau đó nhấn "
        "**Chạy mô hình LSTM**."
    )

    st.markdown(
        """
        ### Hướng dẫn sử dụng
        - Mã mặc định: `VNM.VN`
        - Có thể nhập `AAPL`, `MSFT`, `GOOGL`, `TSLA` hoặc `NVDA`.
        - Chọn khoảng thời gian lấy dữ liệu.
        - Chọn số ngày đầu vào và số epoch huấn luyện.
        - Nhấn **Chạy mô hình LSTM** để xem kết quả.
        """
    )

st.divider()

st.caption(
    "Ứng dụng sử dụng Python, Streamlit, TensorFlow, Keras, "
    "scikit-learn, Pandas, NumPy, Matplotlib và Yahoo Finance."
)
