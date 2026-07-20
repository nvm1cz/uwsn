# S100-S500 best-by-metric figures

Mỗi ảnh đặt `space=100m` và `space=500m` cạnh nhau.
Trục hoành chỉ hiện đúng các số node được xét trong từng space.
Trục tung dùng chung giữa hai panel.

- `01_best_ft5.png`: chọn cấu hình có FT5 cao nhất cho từng thuật toán, space và số node.
- `02_best_residual_energy.png`: chọn cấu hình có năng lượng dư cao nhất tại điểm dừng mô phỏng.
- `03_best_runtime.png`: chọn cấu hình có runtime thấp nhất.
- `04_best_convergence_final.png`: so sánh PSO/GA/DE bằng chi phí tương đối tại iteration cuối; LEACH không có hội tụ.

Các file `best_*_details.csv` ghi rõ mỗi điểm dữ liệu lấy từ folder tham số nào.