import streamlit as st
from nlp_core import classify_sentiment
from database import save_sentiment, load_history
import time # Dùng cho hiệu ứng spinner

st.set_page_config(page_title="Trợ lý Phân loại Cảm xúc Tiếng Việt", layout="wide")

st.title("Trợ lý Phân loại Cảm xúc Tiếng Việt")
st.markdown("Sử dụng **Transformer Pre-trained (PhoBERT)** và lưu trữ bằng **SQLite**.")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Nhập Văn bản Tiếng Việt")
    
    user_input = st.text_area(
        "Nhập câu (tối đa 50 ký tự, có thể viết tắt, thiếu dấu):",
        placeholder="VD: Hôm nay tôi rất vui...", 
        key="user_input", 
        height=150
    )

    if st.button("Phân loại Cảm xúc", type="primary"):
        if not user_input.strip():
            st.error("Vui lòng nhập văn bản để phân loại.")
        else:
            with st.spinner('Đang phân tích...'):
                time.sleep(1) 
                
                result = classify_sentiment(user_input)

                sentiment = result.get('sentiment')
                original_text = result.get('text')
                
                if sentiment == "LENGTH_ERROR":
                    st.error("Câu không hợp lệ! Vui lòng nhập từ 5 đến 50 ký tự.")
                elif sentiment == "PIPELINE_ERROR" or sentiment == "ERROR":
                    st.error("Lỗi Pipeline. Vui lòng kiểm tra console hoặc tên mô hình.")
                else:
                    save_sentiment(original_text, sentiment)
                    st.success(f"Phân loại thành công! Cảm xúc: **{sentiment}**")
                    
                    if sentiment == "POSITIVE":
                        display_sentiment = "TÍCH CỰC (POSITIVE)"
                    elif sentiment == "NEGATIVE":
                        display_sentiment = "TIÊU CỰC (NEGATIVE)"
                    else:
                        display_sentiment = "TRUNG TÍNH (NEUTRAL)"
                        
                    st.subheader(f"Kết quả: {display_sentiment}")
                    st.info(f"Độ tin cậy: {result.get('score', 'N/A'):.4f}")
                    
            st.session_state['refresh_history'] = True


with col2:
    st.header("Lịch sử Phân loại")

    # Khởi tạo số lượng hiển thị mặc định
    if "history_limit" not in st.session_state:
        st.session_state.history_limit = 50

    df_history = load_history(limit=st.session_state.history_limit)

    if not df_history.empty:
        df_history.columns = ['Thời gian', 'Văn bản', 'Cảm xúc']
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("Chưa có lịch sử phân loại nào.")

    if st.button("Tải thêm lịch sử"):
        st.session_state.history_limit += 50
        st.rerun()
