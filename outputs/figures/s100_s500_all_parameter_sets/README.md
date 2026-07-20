# S100-S500 all parameter-set figures

Mỗi thư mục con tương ứng một bộ tham số. Trong từng thư mục:
- `01_ft5.png`: so sánh FT5 giữa các thuật toán.
- `02_residual_energy.png`: so sánh năng lượng dư tại điểm dừng mô phỏng.
- `03_convergence.png`: hội tụ theo iteration nếu file convergence có dữ liệu; nếu không có sẽ ghi rõ trong hình.
- `04_runtime.png`: so sánh thời gian chạy.

Biểu đồ hội tụ so sánh PSO/GA/DE được đặt riêng trong `00_algorithm_convergence_reference` vì LEACH không có vòng lặp tối ưu và file convergence theo input chính không lưu đủ GA/DE cho mọi bộ tham số.

`config_index.csv` là bảng tra cứu folder và tham số đầy đủ.

Các điểm dữ liệu gộp 3 phân phối node và các seed tương ứng trong cùng bộ tham số.