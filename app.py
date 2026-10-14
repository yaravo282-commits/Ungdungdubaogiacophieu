import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

st.set_page_config(
    page_title="Dự báo giá cổ phiếu",
    page_icon="📈",
    layout="wide"
)

st.title("📈 DỰ BÁO GIÁ CỔ PHIẾU BẰNG LSTM")
st.write("Ứng dụng phân tích dữ liệu giá cổ phiếu và dự báo giá trong tương lai.")

st.warning(
    "Dự báo chỉ mang tính tham khảo, không phải lời khuyên đầu tư."
)


@st.cache_data(ttl=3600)
def lay_du_lieu(ma, ngay_bat_dau, ngay_ket_thuc):
    df = yf.download(
        ma,
        start=ngay_bat_dau,
        end=ngay_ket_thuc,
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
    df.columns = ["Gia"]
    df.dropna(inplace=True)

    return df


def tao_du_lieu(data, so_ngay):
    X, y = [], []

    for i in range(so_ngay, len(data)):
        X.append(data[i - so_ngay:i, 0])
        y.append(data[i, 0])

    return np.array(X), np.array(y)


def du_bao_lstm(gia, so_ngay_nhin, so_ngay_du_bao, so_epoch):
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    if len(gia) <= so_ngay_nhin + 20:
        raise ValueError(
            "Không đủ dữ liệu. Hãy chọn khoảng thời gian dài hơn "
            "hoặc giảm số ngày dữ liệu đầu vào."
        )

    vi_tri_chia = int(len(gia) * 0.8)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(gia[:vi_tri_chia])

    gia_scaled = scaler.transform(gia)

    X, y = tao_du_lieu(gia_scaled, so_ngay_nhin)

    vi_tri_chia_mau = vi_tri_chia - so_ngay_nhin

    X_train = X[:vi_tri_chia_mau]
    y_train = y[:vi_tri_chia_mau]
    X_test = X[vi_tri_chia_mau:]
    y_test = y[vi_tri_chia_mau:]

    X_train = X_train.reshape(
        X_train.shape[0], X_train.shape[1], 1
    )
    X_test = X_test.reshape(
        X_test.shape[0], X_test.shape[1], 1
    )

    model = Sequential([
        LSTM(
            32,
            return_sequences=True,
            input_shape=(so_ngay_nhin, 1)
        ),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=so_epoch,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        shuffle=False,
        verbose=0
    )

    y_test_real = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    y_pred_scaled = model.predict(X_test, verbose=0)

    y_pred_real = scaler.inverse_transform(
        y_pred_scaled
    ).flatten()

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = math.sqrt(
        mean_squared_error(y_test_real, y_pred_real)
    )

    du_lieu_cuoi = gia_scaled[-so_ngay_nhin:].flatten().tolist()
    du_bao = []

    for _ in range(so_ngay_du_bao):
        X_moi = np.array(
            du_lieu_cuoi[-so_ngay_nhin:]
        ).reshape(1, so_ngay_nhin, 1)

        gia_moi = float(model.predict(X_moi, verbose=0)[0, 0])

        du_bao.append(gia_moi)
        du_lieu_cuoi.append(gia_moi)

    du_bao = scaler.inverse_transform(
        np.array(du_bao).reshape(-1, 1)
    ).flatten()

    return y_test_real, y_pred_real, mae, rmse, du_bao


with st.sidebar:
    st.header("⚙️ Cài đặt")

    ma_co_phieu = st.text_input(
        "Mã cổ phiếu",
        value="AAPL"
    ).strip().upper()

    ngay_bat_dau = st.date_input(
        "Ngày bắt đầu",
        value=pd.Timestamp.today().date() - pd.Timedelta(days=1825)
    )

    ngay_ket_thuc = st.date_input(
        "Ngày kết thúc",
        value=pd.Timestamp.today().date()
    )

    so_ngay_nhin = st.slider(
        "Số ngày dữ liệu đầu vào",
        min_value=20,
        max_value=100,
        value=60
    )

    so_ngay_du_bao = st.slider(
        "Số ngày muốn dự báo",
        min_value=1,
        max_value=30,
        value=7
    )

    so_epoch = st.slider(
        "Số vòng huấn luyện",
        min_value=5,
        max_value=30,
        value=10
    )

    bam_du_bao = st.button(
        "🚀 Bắt đầu dự báo",
        use_container_width=True
    )


if bam_du_bao:

    if not ma_co_phieu:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    if ngay_bat_dau >= ngay_ket_thuc:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        st.stop()

    with st.spinner("Đang tải dữ liệu cổ phiếu..."):
        try:
            df = lay_du_lieu(
                ma_co_phieu,
                str(ngay_bat_dau),
                str(ngay_ket_thuc + pd.Timedelta(days=1))
            )
        except Exception as e:
            st.error("Không tải được dữ liệu cổ phiếu.")
            st.code(str(e))
            st.stop()

    if df.empty:
        st.error(
            "Không tìm thấy dữ liệu. Hãy kiểm tra mã cổ phiếu "
            "hoặc thử mã AAPL để kiểm tra."
        )
        st.stop()

    st.subheader(f"📊 Dữ liệu cổ phiếu {ma_co_phieu}")

    st.dataframe(
        df.tail(10),
        use_container_width=True
    )

    gia_hien_tai = float(df["Gia"].iloc[-1])

    cot1, cot2 = st.columns(2)

    cot1.metric(
        "Giá đóng cửa gần nhất",
        f"{gia_hien_tai:,.2f}"
    )

    cot2.metric(
        "Số phiên giao dịch",
        f"{len(df):,}"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Gia"],
            mode="lines",
            name="Giá đóng cửa"
        )
    )

    fig.update_layout(
        title=f"Lịch sử giá {ma_co_phieu}",
        xaxis_title="Ngày",
        yaxis_title="Giá",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.spinner("Đang huấn luyện mô hình LSTM..."):
        try:
            gia = df["Gia"].values.reshape(-1, 1)

            y_thuc, y_du_bao_test, mae, rmse, du_bao_tuong_lai = (
                du_bao_lstm(
                    gia,
                    so_ngay_nhin,
                    so_ngay_du_bao,
                    so_epoch
                )
            )

        except ImportError as e:
            st.error(
                "Thiếu thư viện. Hãy kiểm tra file requirements.txt "
                "trên GitHub."
            )
            st.code(str(e))
            st.stop()

        except Exception as e:
            st.error("Không thể huấn luyện hoặc dự báo.")
            st.code(str(e))
            st.stop()

    st.success("Huấn luyện mô hình hoàn tất!")

    st.subheader("📉 Đánh giá mô hình")

    cot1, cot2 = st.columns(2)

    cot1.metric("MAE", f"{mae:,.4f}")
    cot2.metric("RMSE", f"{rmse:,.4f}")

    fig_test = go.Figure()

    fig_test.add_trace(
        go.Scatter(
            y=y_thuc,
            mode="lines",
            name="Giá thực tế"
        )
    )

    fig_test.add_trace(
        go.Scatter(
            y=y_du_bao_test,
            mode="lines",
            name="Giá mô hình dự báo"
        )
    )

    fig_test.update_layout(
        title="So sánh giá thực tế và dự báo trên tập kiểm tra",
        xaxis_title="Phiên giao dịch",
        yaxis_title="Giá",
        template="plotly_white"
    )

    st.plotly_chart(fig_test, use_container_width=True)

    st.subheader("🔮 Kết quả dự báo tương lai")

    ngay_du_bao = pd.bdate_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=so_ngay_du_bao
    )

    bang_du_bao = pd.DataFrame({
        "Ngày": ngay_du_bao.strftime("%d/%m/%Y"),
        "Giá dự báo": np.round(du_bao_tuong_lai, 2)
    })

    st.dataframe(
        bang_du_bao,
        use_container_width=True,
        hide_index=True
    )

    fig_future = go.Figure()

    fig_future.add_trace(
        go.Scatter(
            x=df.index[-90:],
            y=df["Gia"].iloc[-90:],
            mode="lines",
            name="Giá lịch sử"
        )
    )

    fig_future.add_trace(
        go.Scatter(
            x=ngay_du_bao,
            y=du_bao_tuong_lai,
            mode="lines+markers",
            name="Giá dự báo",
            line=dict(dash="dash")
        )
    )

    fig_future.update_layout(
        title=f"Dự báo giá cổ phiếu {ma_co_phieu}",
        xaxis_title="Ngày",
        yaxis_title="Giá",
        template="plotly_white"
    )

    st.plotly_chart(fig_future, use_container_width=True)

    csv = bang_du_bao.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "📥 Tải kết quả dự báo CSV",
        data=csv,
        file_name=f"du_bao_{ma_co_phieu}.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Nhập mã cổ phiếu ở thanh bên trái rồi nhấn "
        "**Bắt đầu dự báo**."
    )

    st.markdown(
        """
        ### 📌 Hướng dẫn
        - `AAPL`: Apple
        - `MSFT`: Microsoft
        - `GOOGL`: Alphabet
        - `TSLA`: Tesla
        - `NVDA`: NVIDIA
        - `VNM.VN`: mã VNM trên Yahoo Finance, nếu có dữ liệu

        Chọn khoảng thời gian, số ngày đầu vào và số ngày dự báo,
        sau đó nhấn **Bắt đầu dự báo**.
        """
    )

st.divider()

st.caption(
    "Mô hình LSTM chỉ dùng cho mục đích học tập và tham khảo. "
    "Kết quả không đảm bảo giá cổ phiếu trong tương lai."
)
