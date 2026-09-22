

Dựa theo yêu cầu tối thiểu của M1 Draft trong handbook, dưới đây là kế hoạch từng bước cụ thể:
**Bước 1: Chuẩn bị Dữ liệu và Phân tích Khám phá (EDA - Khối lượng: ~2-3 ngày)**
**Tải dữ liệu:** Tải bộ dữ liệu **Fashion-MNIST** làm bộ dữ liệu chính để báo cáo (có thể tải thêm MNIST chỉ để test code/debug nếu muốn).

**Thực hiện EDA (Bắt buộc):** Trực quan hóa dữ liệu. Đảm bảo EDA của bạn có đủ 4 yếu tố:
1. Phân bố lớp (class distribution).
2. Kích thước đầu vào (input size).
3. Phân tích sự mất cân bằng dữ liệu (imbalance analysis).
4. Hiển thị một số hình ảnh mẫu đại diện cho các lớp (representative samples).

**Xây dựng Dataset & DataLoader:**
* Chia tập dữ liệu (train/validation/test) với seed cố định (Seed: 42) để đảm bảo công bằng khi so sánh các mô hình sau này.
* Viết code PyTorch `Dataset` và `DataLoader`, thiết lập các phép tiền xử lý (preprocessing) cơ bản (chuyển sang tensor, chuẩn hóa với $\mu=0.2860, \sigma=0.3530$).

> [!NOTE]
> **Đã hoàn thành sinh 5 biểu đồ EDA chuẩn báo cáo tại thư mục [`./eda_figures/`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/eda_figures):**
> 1. [`fig1_representative_samples.png`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/eda_figures/fig1_representative_samples.png): Lưới 10 lớp x 5 mẫu đại diện (độ phân giải $28 \times 28$, grayscale).
> 2. [`fig2_class_distribution.png`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/eda_figures/fig2_class_distribution.png): Biểu đồ cột phân bố lớp Train (50k - 5k/lớp), Val (10k - 1k/lớp), Test (10k - 1k/lớp) chứng minh cân bằng tuyệt đối (Zero Imbalance, tỷ lệ 1:1).
> 3. [`fig3_pixel_distribution.png`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/eda_figures/fig3_pixel_distribution.png): Phân tích phân bố cường độ pixel thô [0, 255] và sau chuẩn hóa, chứng minh độ thưa nền (~52% pixel bằng 0).
> 4. [`fig4_class_prototypes.png`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/eda_figures/fig4_class_prototypes.png): Ảnh nguyên mẫu trung bình ($\mu_c$) và bản đồ nhiệt độ lệch chuẩn ($\sigma_c$) thể hiện tính đối xứng trục đứng và vùng biến thiên cao.
> 5. [`fig5_pca_clusters.png`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/eda_figures/fig5_pca_clusters.png): Giảm chiều 2D PCA thể hiện mức độ tách cụm giữa nhóm giày dép (Sandal, Sneaker, Ankle boot) và nhóm áo quần (T-shirt, Shirt, Pullover, Coat).
> 
> *Script tái hiện:* `python generate_eda_graphs.py`



**Bước 2: Xây dựng Pipeline Huấn luyện (Train/Val Loop - Khối lượng: ~2 ngày)**
 **Vòng lặp huấn luyện:** Viết các hàm `train_epoch()` và `validate_epoch()` bằng PyTorch.
 **Tính toán Loss & Metrics:** Đảm bảo mã nguồn tính được hàm mất mát (Loss) và các chỉ số bắt buộc như **Accuracy** và **Macro-F1**.
 **Theo dõi & Lưu trữ:** Code phải in ra hoặc lưu lại lịch sử huấn luyện (loss/metric curves) để phục vụ cho báo cáo sau này.

> [!NOTE]
> **Đã hoàn thành đầy đủ Bước 2 tại [`engine.py`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/engine.py) và [`metrics.py`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/metrics.py):**
> 1. **Vòng lặp Train/Val:** `train_epoch()` và `validate_epoch()` tối ưu trên GPU CUDA với non-blocking transfer.
> 2. **Tính toán các chỉ số bắt buộc (Mục 3):**
>    - `CrossEntropyLoss`: Tính toán trung bình chính xác theo số mẫu.
>    - `Accuracy`: Độ chính xác tổng quát phân loại trên 10 lớp.
>    - `Macro-F1 Score`: Tính F1 không trọng số trên cả 10 lớp theo đúng rubric handbook (`sklearn.metrics.f1_score(..., average='macro')`).
>    - `compute_metrics()` & `plot_confusion_matrix()`: Xuất bảng chi tiết Precision/Recall/F1 cho từng lớp và sinh biểu đồ nhiệt Ma trận nhầm lẫn (`confusion_matrix.png`).
> 3. **Theo dõi & Lưu trữ Lịch sử Huấn luyện (Mục 4):**
>    - Lưu lịch sử từng epoch ra cả file JSON (`history.json`) và CSV (`history.csv`) để dễ dàng vẽ biểu đồ hoặc chèn vào báo cáo LaTeX/Word.
>    - `EarlyStopping`: Dừng sớm nếu Validation Macro-F1 không cải thiện sau $N$ epoch (mặc định: 3 epoch) và tự động khôi phục trọng số tối ưu nhất.
>    - Tự động lưu checkpoint mô hình tốt nhất (`checkpoints/best_{model_name}.pth`).
>    - `plot_training_history()`: Xuất biểu đồ chuẩn học thuật 3 panel (`learning_curves.png`) thể hiện đồng thời Loss, Accuracy, và Macro-F1 curves qua các epoch.
> - *Lệnh chạy kiểm thử:* `python engine.py` (Đã pass 100% trên GPU NVIDIA RTX 3060).




**Bước 3: Triển khai 2 Mô hình Cơ bản (Linear & MLP - Khối lượng: ~2-3 ngày)**
Ở cột mốc M1, bạn chỉ bắt buộc chạy được 2 mô hình (CNN là tùy chọn, chưa bắt buộc):
 **Mô hình 1: Linear / Softmax Classifier:**
 Thiết kế: `Flatten` ảnh đầu vào thành vector $\rightarrow$ qua một lớp Tuyến tính (Linear layer) $\rightarrow$ cho ra logits.
 *Lưu ý quan trọng:* Sử dụng hàm mất mát `CrossEntropyLoss` của PyTorch và **tuyệt đối không** dùng hàm softmax trước lớp loss này.

 **Mô hình 2: Multilayer Perceptron (MLP):**
 Thiết kế: `Flatten` ảnh $\rightarrow$ qua 2 lớp ẩn ($784 \rightarrow 256 \rightarrow 128 \rightarrow 10$) $\rightarrow$ Lớp đầu ra.
 Kỹ thuật áp dụng: Hàm kích hoạt `ReLU`, điều chuẩn `Dropout(p=0.2)` và `Weight Decay (L2 = 1e-4)`.

> [!NOTE]
> **Đã hoàn thành Bước 3 tại file [`models.py`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/models.py):**
> - `LinearClassifier`: Đúng 7,850 tham số ($784 \times 10 + 10$).
> - `MLPClassifier`: Đúng 235,146 tham số ($(784 \times 256 + 256) + (256 \times 128 + 128) + (128 \times 10 + 10)$) khớp 100% tài liệu Group Goat.
> - *Kiểm thử tham số & forward pass:* `python models.py` (Đã pass 100%).

**Bước 4: Chạy Code, Gỡ lỗi và Lưu kết quả (Khối lượng: ~2 ngày)**
 Chạy pipeline huấn luyện cho cả Linear và MLP trên bộ Fashion-MNIST (50,000 train, 10,000 val, 10,000 test).
 Theo dõi Loss và Metrics. Nếu code gặp lỗi (bug) hoặc mô hình không học được, hãy dùng bộ MNIST nhỏ gọn để gỡ lỗi nhanh.
 Lưu lại các Checkpoint của mô hình và trích xuất dữ liệu biểu đồ (train/val curves) để đưa vào Draft Report.

> [!NOTE]
> **Đã hoàn thành xuất sắc Bước 4 — Huấn luyện thật 100% trên 50,000 ảnh Fashion-MNIST bằng GPU RTX 3060:**
>
> #### Bảng so sánh kết quả định lượng chính thức (Test Set 10,000 ảnh chưa từng thấy):
> | Kiến trúc Mô hình | Số lượng Tham số | Test Loss | Test Accuracy | Test Macro-F1 (Rubric) | Thời gian Train |
> | :--- | :---: | :---: | :---: | :---: | :---: |
> | **Linear / Softmax** | $7,850$ | $0.4392$ | **$84.49\%$** | **$84.37\%$** | $200.4\text{s}$ |
> | **MLP Classifier** | $235,146$ | $0.3122$ | **$88.86\%$** | **$88.83\%$** | $197.3\text{s}$ |
>
> *Độ cải thiện của MLP so với Linear:* $+4.37\%$ Accuracy, $+4.46\%$ Macro-F1, giảm $-0.1270$ Test Loss.
>
> #### Các deliverables đã xuất vào thư mục [`./outputs/`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/outputs):
> 1. Checkpoint tốt nhất: [`best_linear.pth`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/outputs/checkpoints/best_linear.pth), [`best_mlp.pth`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/outputs/checkpoints/best_mlp.pth)
> 2. Lịch sử chi tiết: [`linear_history.csv`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/outputs/checkpoints/linear_history.csv), [`mlp_history.csv`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/outputs/checkpoints/mlp_history.csv)
> 3. Biểu đồ đường học tập: [`learning_curves_linear.png`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/outputs/figures/learning_curves_linear.png), [`learning_curves_mlp.png`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/outputs/figures/learning_curves_mlp.png)
> 4. Ma trận nhầm lẫn: [`confusion_matrix_linear.png`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/outputs/figures/confusion_matrix_linear.png), [`confusion_matrix_mlp.png`](file:///c:/Users/huymi/OneDrive/Tài%20liệu/HCMUT/DeepLearning/outputs/figures/confusion_matrix_mlp.png)



**Bước 5: Soạn Báo cáo Nháp và Nộp bài (Draft Report & Submission - Khối lượng: ~2 ngày)**
Vì đây là Draft, báo cáo chưa cần hoàn hảo nhưng phải có các phần sau:

 **Cập nhật Repository:** Đẩy toàn bộ source code (EDA, DataLoader, Train loop, Models) lên GitHub. Đảm bảo có hướng dẫn chạy (README.md).


 **Viết Báo cáo Nháp (Phần 1 & 2):**
 Mô tả Dữ liệu: Trình bày kết quả EDA, cách chia dữ liệu, và tiền xử lý.
 Phương pháp luận: Vẽ sơ đồ pipeline, mô tả kiến trúc mô hình Linear và MLP, ghi chú các thiết lập huấn luyện (optimizer, learning rate, batch size, v.v.).
 **Kê khai AI (AI Usage Disclosure):** Đừng quên cập nhật file `AI_USAGE.md` và thêm phần kê khai vào báo cáo/trang web nếu nhóm có dùng AI (nếu không dùng thì ghi rõ là không dùng).

**Nộp bài lên LMS:** Cập nhật đường link trang web và repo của nhóm lên hệ thống LMS trước thời hạn.
Các mô hình phức tạp như CNN, LSTM/GRU và Transformer, cùng với các so sánh chuyên sâu và video thuyết trình có thể để dành thực hiện cho giai đoạn **M2 Final (21/10/2026)**.
