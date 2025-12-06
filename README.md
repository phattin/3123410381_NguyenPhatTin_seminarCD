Trợ lý Phân Loại Cảm Xúc Tiếng Việt (PhoBERT + Streamlit + SQLite)

Các bước để run project

Bước 1: Cài đặt Python ( nếu chưa cài )
Cài đặt Python 3.10 hoặc 3.11 ( https://www.python.org/downloads/windows/ )
Lưu ý: Không dùng được Python 3.12+
Quan trọng: Nhớ tick vào Add python.exe to PATH

Bước 2: Tạo và kích hoạt môi trường ảo, nhập trong Terminal
python -m venv venv

Bước 3: Cài đặt các thư viện cần thiết, nhập trong Terminal
pip install streamlit transformers torch underthesea sentencepiece pandas

Bước 4: Chạy ứng dụng Streamlit, nhập trong Terminal
streamlit run app.py
(Nhập email nếu có yêu cầu)
Truy cập tại địa chỉ: http://localhost:8501