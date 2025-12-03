## **Lab 01: Scraping**

### **1. Mô tả đồ án**
- **Họ tên:** Trương Quang Phát
- **MSSV:** 23120318
- **Môn:** Nhập môn Khoa học dữ liệu
- **Lớp:** CQ2023/21

### **2. Mô tả đồ án**
Đồ án này thực hiện một trình cào dữ liệu (scraper) để thu thập dữ liệu từ kho lưu trữ mở arXiv, theo các yêu cầu được mô tả trong đề bài.
Mục tiêu chính của scraper là:
  - Thu thập một tập các bài báo arXiv dựa trên một dải ID được chỉ định (ví dụ: từ `2409.05017` đến `2409.10016`).
  - Đối với mỗi ID bài báo, hệ thống sẽ cào tất cả các phiên bản (versions) có sẵn của bài báo đó, từ `v1` đến phiên bản mới nhất.
  - Tải về tệp nguồn (source files) của mỗi phiên bản. Chương trình có khả năng xử lý cả định dạng lưu trữ `.tar.gz` và các tệp nén `.gz` đơn lẻ (chứa TeX/BibTeX).
  - Trích xuất và giữ lại các tệp nguồn LaTeX (`.tex`) và BibTeX (`.bib`), đồng thời loại bỏ tất cả các tệp khác (như hình ảnh, file PDF,...) để giảm dung lượng lưu trữ, theo yêu cầu của đề bài.
  - Thu thập và lưu trữ siêu dữ liệu (metadata) của bài báo (như tiêu đề, tác giả, ngày nộp, ngày sửa đổi) vào tệp `metadata.json`.
  - Sử dụng API của Semantic Scholar để lấy thông tin về các tài liệu tham khảo (references) của bài báo. Các tài liệu tham khảo có ID arXiv sẽ được lọc và lưu vào tệp `references.json`.

Chương trình được thiết kế để tối ưu hiệu suất bằng cách sử dụng:
  - **Xử lý song song:** Sử dụng `ThreadPoolExecutor` (luồng) để tải xuống đồng thời nhiều phiên bản bài báo (I/O-bound) và `ProcessPoolExecutor` (tiến trình) để giải nén và lọc tệp (CPU-bound).
  - **Xử lý theo lô (Batching):** Gửi yêu cầu hàng loạt đến API Semantic Scholar (tối đa 100-200 ID mỗi lần) để giảm độ trễ mạng và tăng thông lượng.
  - **Cơ chế "Fast Mode":** Tự động điều chỉnh thời gian chờ (wait time) giữa các yêu cầu API xuống mức tối thiểu và tăng số lượng workers để tối đa hóa tốc độ cào dữ liệu, đặc biệt hữu ích khi chạy trên các môi trường như Colab.

### **3. Cấu trúc thư mục**
**Cấu trúc mã nguồn**
Theo yêu cầu của đề bài, mã nguồn được tổ chức trong thư mục `src/`, cùng với các tệp báo cáo và yêu cầu môi trường:
```c
23120318/
|-- src/
|    |-- scraper.py
|    |-- stats.py
|    |-- requirements.txt
|-- README.md
|-- Report.docx
```
**Cấu trúc dữ liệu đầu ra**
Chương trình tạo ra một thư mục đầu ra `23120318` với cấu trúc dữ liệu tuân thủ nghiêm ngặt yêu cầu của đề bài.
```c
<23120318>/
|-- <yymm-id>/
|   |-- tex/
|   |   |-- <yymm-id>v<version>/
|   |   |   |-- *.tex
|   |   |   |-- *.bib
|   |   |-- <subfolders>
|   |   |       |-- *.tex
|   |   |       |-- *.bib
|   |   |       |-- ... <recursively following the structure of original TeX sources>
|   |-- metadata.json
|   |-- references.json
|-- ...
```

### **4. Yêu cầu môi trường**
Chương trình yêu cầu Python 3.x. Các thư viện bên ngoài cần thiết được liệt kê trong tệp `requirements.txt`.
Cài đặt tất cả các thư viện cần thiết bằng lệnh:
```c
pip install -r requirements.txt
```

### **5. Thực thi mã nguồn**
Chương trình được thực thi thông qua dòng lệnh (CLI) từ tệp `src/scraper.py`.
#### **5.1. Thiết lập các tham số tùy chọn**
  - `--outdir` / `-o`: Thư mục đầu ra để lưu dữ liệu (Mặc định: `23120318`).
  - `--year-month`: Chỉ số năm và tháng cho arXiv ID (Mặc định: 2409)
  - `--start`: Chỉ số ID bắt đầu (Mặc định: `5017`, tương ứng với `2409.05017`).
  - `--end`: Chỉ số ID kết thúc (Mặc định: `10016`, tương ứng với `2409.10016`).
  - `--download-workers`: Số luồng tải xuống song song (Tự động điều chỉnh dựa trên số CPU và "fast mode").
  - `--decompress-workers`: Số tiến trình giải nén song song (Tự động điều chỉnh dựa trên số CPU và "fast mode").
  - `--no-prefetch-s2`: Tắt tính năng tìm nạp trước (prefetch) S2 theo lô. Thay vào đó, chương trình sẽ gọi API S2 cho từng bài báo một cách tuần tự.
  - `--no-fast-mode`: Tắt "Fast Mode". Khi tắt, chương trình sẽ chạy ở chế độ polite với thời gian chờ giữa các yêu cầu API lâu hơn (ví dụ: `ARXIV_WAIT = 3.1s`, `S2_MIN_WAIT = 1.1s`), giảm số lượng workers. Mặc định, "Fast Mode" được bật.
  - `--arxiv-wait` / `--s2-wait` / v.v.: Ghi đè thủ công thời gian chờ cụ thể giữa các yêu cầu API.

#### **5.2. Cách thực thi**
- **Di chuyển vào thư mục `src`**
    ```c
    cd src
    ```
- **Tạo môi trường ảo (virtual environment)**
    ```c
    python -m venv my_env
    ```
- **Kích hoạt virtual environment**
    ```c
    my_env\Scripts\activate
    ```
- **Cài đặt các thư viện trong `requirements.txt`**
    ```c
    pip install -r requirements.txt
    ```
- Thực thi tệp `scraper.py`
    **Chạy mặc định (Fast Mode):** Chạy với dải ID mặc định (`2409.05017` đến `2409.10016`) và lưu vào thư mục `../23120318`. Chương trình sẽ chạy ở chế độ nhanh (fast mode) với thời gian chờ API ngắn và nhiều workers.
    ```c
    python scraper.py
    ```
    **Chạy ở chế độ Polite (chậm và an toàn):** Sử dụng cờ `--no-fast-mode` để tuân thủ thời gian chờ dài hơn giữa các yêu cầu API.
    ```c
    python scraper.py --no-fast-mode
    ```

    **Chạy với dải ID tùy chỉnh:** Ví dụ: cào 10 bài báo đầu tiên và lưu vào thư mục my_test.
    ```c
    python scraper.py --start 5017 --end 5026 --outdir ../my_test
    ```

    **Chạy với số workers tùy chỉnh:**
    ```c
    python scraper.py --download-workers 16 --decompress-workers 8
    ```

#### **5.3. Output**
- **Dữ liệu:** Chương trình sẽ tạo và điền dữ liệu vào thư mục đầu ra (--outdir) theo cấu trúc đã mô tả ở Mục 3.
- **Log (stdout):** Tiến trình cào dữ liệu được hiển thị trực tiếp trên terminal bằng thanh tiến trình `tqdm`, bao gồm tiến trình tìm nạp S2 và tiến trình xử lý từng bài báo. Các lỗi hoặc cảnh báo (ví dụ: bài báo không tìm thấy, lỗi mạng) cũng sẽ được in ra.
- **Thống kê:** Sau khi hoàn tất, một tệp `report_stats.json` sẽ được tạo ra bên ngoài thư mục đầu ra (trong thư mục cha của nó). Tệp này chứa các thống kê tổng hợp về hiệu suất theo yêu cầu của đề bài (ví dụ: tổng thời gian, số bài báo thành công, kích thước tệp trung bình, số tham khảo trung bình, RAM tối đa sử dụng, v.v.).

### **6. Tham khảo**
**[1]** arXiv: https://arxiv.org/
**[2]** Attention Is All You Need. https://arxiv.org/abs/1706.03762/
**[3]** Semantic Scholar API: https://api.semanticscholar.org/

### **7. Video demo**
**Link video**: https://www.youtube.com/watch?v=xn5EPZkHe0Q