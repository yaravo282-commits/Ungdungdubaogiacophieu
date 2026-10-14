import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import math
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Dự báo giá cổ phiếu bằng LSTM",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Ứng dụng dự báo giá cổ phiếu bằng LSTM")
st.write(
    "Ứng dụng sử dụng mô hình học sâu LSTM để phân tích dữ liệu "
    "giá cổ phiếu trong quá khứ và dự báo giá trong tương lai."
)

st.warning(
    "Lưu ý: Đây là dự báo tham khảo từ mô hình học máy, "
    "không phải lời khuyên đầu tư. Giá thực tế có thể khác đáng kể."
)

@st.cache_data(ttl=3600)
def tai_du_lieu(ma_co_phieu, ngay_bat_dau, ngay_ket_thuc):
    du_lieu = yf.download(
        ma_co_phieu,
        start=ngay_bat_dau,
        end=ngay_ket_thuc,
        auto_adjust=True,
        progress=False
    )

    if du_lieu.empty:
        return pd.DataFrame()

    if isinstance(du_lieu.columns, pd.MultiIndex):
        du_lieu.columns = du_lieu.columns.get_level_values(0)

    du_lieu = du_lieu[["Close"]].copy()
    du_lieu.columns = ["Gia_dong_cua"]
    du_lieu = du_lieu.dropna()

    return du_lieu


def tao_du_lieu_lstm(du_lieu, so_ngay_nhin_truoc):
    X = []
    y = []

    for i in range(so_ngay_nhin_truoc, len(du_lieu)):
        X.append(du_lieu[i - so_ngay_nhin_truoc:i, 0])
        y.append(du_lieu[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y


def tao_mo_hinh_lstm(so_ngay_nhin_truoc):
    mo_hinh = Sequential([
        LSTM(
            64,
            return_sequences=True,
            input_shape=(so_ngay_nhin_truoc, 1)
        ),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    mo_hinh.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return mo_hinh


def du_bao_tuong_lai(
    mo_hinh,
    du_lieu_scaled,
    scaler,
    so_ngay_nhin_truoc,
    so_ngay_du_bao
):
    du_lieu_cuoi = du_lieu_scaled[-so_ngay_nhin_truoc:].copy()
    ket_qua_du_bao = []

    for _ in range(so_ngay_du_bao):
        X_moi = du_lieu_cuoi.reshape(1, so_ngay_nhin_truoc, 1)

        gia_du_bao_scaled = mo_hinh.predict(
            X_moi,
            verbose=0
        )[0, 0]

        ket_qua_du_bao.append(gia_du_bao_scaled)

        du_lieu_cuoi = np.append(
            du_lieu_cuoi[1:],
            [[gia_du_bao_scaled]],
            axis=0
        )

    ket_qua_du_bao = np.array(ket_qua_du_bao).reshape(-1, 1)

    return scaler.inverse_transform(ket_qua_du_bao).flatten()


with st.sidebar:
    st.header("⚙️ Cài đặt")

    ma_co_phieu = st.text_input(
        "Mã cổ phiếu",
        value="AAPL",
        help="Ví dụ: AAPL, MSFT, GOOGL, TSLA, NVDA, VNM.VN"
    ).strip().upper()

    ngay_bat_dau = st.date_input(
        "Ngày bắt đầu dữ liệu",
        value=pd.Timestamp.today().date() - pd.Timedelta(days=1825)
    )

    ngay_ket_thuc = st.date_input(
        "Ngày kết thúc dữ liệu",
        value=pd.Timestamp.today().date()
    )

    so_ngay_nhin_truoc = st.slider(
        "Số ngày dữ liệu dùng để dự báo",
        min_value=20,
        max_value=120,
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
        max_value=50,
        value=20
    )

    chay_du_bao = st.button(
        "🚀 Chạy dự báo",
        use_container_width=True
    )


if chay_du_bao:
    if not ma_co_phieu:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    if ngay_bat_dau >= ngay_ket_thuc:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        st.stop()

    with st.spinner("Đang tải dữ liệu cổ phiếu..."):
        try:
            du_lieu = tai_du_lieu(
                ma_co_phieu,
                str(ngay_bat_dau),
                str(ngay_ket_thuc + pd.Timedelta(days=1))
            )
        except Exception as e:
            st.error(f"Không thể tải dữ liệu: {e}")
            st.stop()

    if du_lieu.empty:
        st.error(
            "Không tìm thấy dữ liệu. Hãy kiểm tra mã cổ phiếu "
            "hoặc thử một mã khác."
        )
        st.stop()

    if len(du_lieu) < so_ngay_nhin_truoc + 30:
        st.error(
            "Dữ liệu không đủ để huấn luyện mô hình. "
            "Hãy chọn khoảng thời gian dài hơn."
        )
        st.stop()

    st.subheader(f"📊 Dữ liệu cổ phiếu {ma_co_phieu}")

    gia_hien_tai = float(du_lieu["Gia_dong_cua"].iloc[-1])
    gia_truoc_do = float(du_lieu["Gia_dong_cua"].iloc[-2])

    thay_doi = gia_hien_tai - gia_truoc_do
    phan_tram_thay_doi = (
        thay_doi / gia_truoc_do * 100
        if gia_truoc_do != 0
        else 0
    )

    cot1, cot2, cot3 = st.columns(3)

    cot1.metric(
        "Giá đóng cửa gần nhất",
        f"{gia_hien_tai:,.2f}",
        f"{thay_doi:+,.2f} ({phan_tram_thay_doi:+.2f}%)"
    )

    cot2.metric(
        "Số phiên giao dịch",
        f"{len(du_lieu):,}"
    )

    cot3.metric(
        "Ngày dữ liệu gần nhất",
        du_lieu.index[-1].strftime("%d/%m/%Y")
    )

    bieu_do_gia = go.Figure()

    bieu_do_gia.add_trace(
        go.Scatter(
            x=du_lieu.index,
            y=du_lieu["Gia_dong_cua"],
            mode="lines",
            name="Giá đóng cửa"
        )
    )

    bieu_do_gia.update_layout(
        title=f"Lịch sử giá cổ phiếu {ma_co_phieu}",
        xaxis_title="Ngày",
        yaxis_title="Giá",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(
        bieu_do_gia,
        use_container_width=True
    )

    st.subheader("🧠 Huấn luyện mô hình LSTM")

    tien_do = st.progress(0)
    thong_bao = st.empty()

    try:
        gia = du_lieu["Gia_dong_cua"].values.reshape(-1, 1)

        vi_tri_chia = int(len(gia) * 0.8)

        if vi_tri_chia <= so_ngay_nhin_truoc:
            st.error(
                "Không đủ dữ liệu huấn luyện. "
                "Hãy giảm số ngày nhìn trước hoặc tăng khoảng thời gian dữ liệu."
            )
            st.stop()

        scaler = MinMaxScaler(feature_range=(0, 1))

        scaler.fit(gia[:vi_tri_chia])

        gia_scaled = scaler.transform(gia)

        X, y = tao_du_lieu_lstm(
            gia_scaled,
            so_ngay_nhin_truoc
        )

        vi_tri_chia_mau = vi_tri_chia - so_ngay_nhin_truoc

        X_train = X[:vi_tri_chia_mau]
        y_train = y[:vi_tri_chia_mau]

        X_test = X[vi_tri_chia_mau:]
        y_test = y[vi_tri_chia_mau:]

        if len(X_train) == 0 or len(X_test) == 0:
            st.error(
                "Không đủ dữ liệu để chia tập huấn luyện và kiểm tra."
            )
            st.stop()

        mo_hinh = tao_mo_hinh_lstm(
            so_ngay_nhin_truoc
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        lich_su = mo_hinh.fit(
            X_train,
            y_train,
            epochs=so_epoch,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping],
            shuffle=False,
            verbose=0
        )

        tien_do.progress(100)
        thong_bao.success("Huấn luyện mô hình hoàn tất!")

        st.subheader("📉 Đánh giá mô hình")

        y_test_thuc = scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        y_test_du_bao_scaled = mo_hinh.predict(
            X_test,
            verbose=0
        )

        y_test_du_bao = scaler.inverse_transform(
            y_test_du_bao_scaled
        ).flatten()

        mae = mean_absolute_error(
            y_test_thuc,
            y_test_du_bao
        )

        rmse = math.sqrt(
            mean_squared_error(
                y_test_thuc,
                y_test_du_bao
            )
        )

        cot1, cot2 = st.columns(2)

        cot1.metric(
            "MAE",
            f"{mae:,.4f}"
        )

        cot2.metric(
            "RMSE",
            f"{rmse:,.4f}"
        )

        st.caption(
            "MAE và RMSE đo sai số dự báo trên tập kiểm tra. "
            "Giá trị thấp hơn thường cho thấy sai số nhỏ hơn, "
            "nhưng không đảm bảo dự báo tương lai chính xác."
        )

        bieu_do_so_sanh = go.Figure()

        ngay_test = du_lieu.index[
            vi_tri_chia:
        ]

        bieu_do_so_sanh.add_trace(
            go.Scatter(
                x=ngay_test,
                y=y_test_thuc,
                mode="lines",
                name="Giá thực tế"
            )
        )

        bieu_do_so_sanh.add_trace(
            go.Scatter(
                x=ngay_test,
                y=y_test_du_bao,
                mode="lines",
                name="Giá dự báo"
            )
        )

        bieu_do_so_sanh.update_layout(
            title="So sánh giá thực tế và giá dự báo trên tập kiểm tra",
            xaxis_title="Ngày",
            yaxis_title="Giá",
            template="plotly_white",
            hovermode="x unified"
        )

        st.plotly_chart(
            bieu_do_so_sanh,
            use_container_width=True
        )

        st.subheader(
            f"🔮 Dự báo {so_ngay_du_bao} ngày giao dịch tiếp theo"
        )

        gia_du_bao_tuong_lai = du_bao_tuong_lai(
            mo_hinh=mo_hinh,
            du_lieu_scaled=gia_scaled,
            scaler=scaler,
            so_ngay_nhin_truoc=so_ngay_nhin_truoc,
            so_ngay_du_bao=so_ngay_du_bao
        )

        ngay_cuoi = du_lieu.index[-1]

        ngay_du_bao = pd.bdate_range(
            start=ngay_cuoi + pd.Timedelta(days=1),
            periods=so_ngay_du_bao
        )

        bang_du_bao = pd.DataFrame({
            "Ngày dự báo": ngay_du_bao,
            "Giá dự báo": gia_du_bao_tuong_lai
        })

        bang_du_bao["Ngày dự báo"] = (
            bang_du_bao["Ngày dự báo"].dt.strftime("%d/%m/%Y")
        )

        bang_du_bao["Giá dự báo"] = (
            bang_du_bao["Giá dự báo"].round(2)
        )

        st.dataframe(
            bang_du_bao,
            use_container_width=True,
            hide_index=True
        )

        gia_cuoi_cung = float(
            gia_du_bao_tuong_lai[-1]
        )

        thay_doi_du_bao = gia_cuoi_cung - gia_hien_tai

        phan_tram_du_bao = (
            thay_doi_du_bao / gia_hien_tai * 100
            if gia_hien_tai != 0
            else 0
        )

        cot1, cot2, cot3 = st.columns(3)

        cot1.metric(
            "Giá hiện tại",
            f"{gia_hien_tai:,.2f}"
        )

        cot2.metric(
            "Giá dự báo cuối kỳ",
            f"{gia_cuoi_cung:,.2f}"
        )

        cot3.metric(
            "Mức thay đổi dự báo",
            f"{thay_doi_du_bao:+,.2f}",
            f"{phan_tram_du_bao:+.2f}%"
        )

        lich_su_hien_thi = du_lieu.tail(90)

        bieu_do_tuong_lai = go.Figure()

        bieu_do_tuong_lai.add_trace(
            go.Scatter(
                x=lich_su_hien_thi.index,
                y=lich_su_hien_thi["Gia_dong_cua"],
                mode="lines",
                name="Giá lịch sử"
            )
        )

        bieu_do_tuong_lai.add_trace(
            go.Scatter(
                x=ngay_du_bao,
                y=gia_du_bao_tuong_lai,
                mode="lines+markers",
                name="Giá dự báo",
                line=dict(dash="dash")
            )
        )

        bieu_do_tuong_lai.update_layout(
            title=f"Dự báo giá cổ phiếu {ma_co_phieu}",
            xaxis_title="Ngày",
            yaxis_title="Giá",
            template="plotly_white",
            hovermode="x unified"
        )

        st.plotly_chart(
            bieu_do_tuong_lai,
            use_container_width=True
        )

        csv_du_bao = bang_du_bao.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label="📥 Tải kết quả dự báo CSV",
            data=csv_du_bao,
            file_name=f"du_bao_{ma_co_phieu}.csv",
            mime="text/csv",
            use_container_width=True
        )

        with st.expander("📚 Xem lịch sử huấn luyện"):
            bieu_do_loss = go.Figure()

            bieu_do_loss.add_trace(
                go.Scatter(
                    y=lich_su.history["loss"],
                    mode="lines",
                    name="Loss huấn luyện"
                )
            )

            if "val_loss" in lich_su.history:
                bieu_do_loss.add_trace(
                    go.Scatter(
                        y=lich_su.history["val_loss"],
                        mode="lines",
                        name="Loss kiểm định"
                    )
                )

            bieu_do_loss.update_layout(
                title="Quá trình huấn luyện mô hình",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                template="plotly_white"
            )

            st.plotly_chart(
                bieu_do_loss,
                use_container_width=True
            )

        st.success("Đã hoàn thành dự báo!")

    except Exception as e:
        st.error(
            "Đã xảy ra lỗi trong quá trình huấn luyện hoặc dự báo."
        )
        st.exception(e)

else:
    st.info(
        "👈 Hãy chọn mã cổ phiếu và các thông số ở thanh bên, "
        "sau đó nhấn **Chạy dự báo**."
    )

    st.subheader("📌 Hướng dẫn sử dụng")

    st.markdown(
        """
        1. Nhập mã cổ phiếu, ví dụ `AAPL`, `MSFT`, `GOOGL`, `TSLA` hoặc `NVDA`.
        2. Chọn khoảng thời gian lấy dữ liệu lịch sử.
        3. Chọn số ngày dữ liệu quá khứ dùng để dự báo.
        4. Chọn số ngày muốn dự báo và số vòng huấn luyện.
        5. Nhấn **Chạy dự báo** để bắt đầu.
        6. Xem biểu đồ, chỉ số sai số và tải kết quả CSV nếu cần.

        **Lưu ý:** Ngày dự báo được tính theo ngày làm việc thông thường,
        chưa loại trừ các ngày nghỉ lễ của từng thị trường.
        Dự báo LSTM chỉ mang tính tham khảo, không đảm bảo giá thực tế.
        """
    )

st.divider()

st.caption(
    "Ứng dụng minh họa sử dụng Python, Streamlit, Yahoo Finance, "
    "TensorFlow và mô hình LSTM."
)
