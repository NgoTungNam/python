# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Giả lập dữ liệu sinh viên (bạn có thể thay thế bằng dữ liệu thực tế của bạn)
np.random.seed(42)

# Giả sử dữ liệu có 7 đặc điểm, trong đó có điểm học tập (trung bình học tập từ 0 đến 10)
# và các hoạt động ngoại khóa (tham gia các hoạt động, câu lạc bộ, thể thao,...)
n_students = 100
data = {
    'Điểm học tập': np.random.uniform(5, 10, n_students),  # Điểm học tập từ 5 đến 10
    'Tham gia CLB': np.random.randint(0, 2, n_students),    # 0 hoặc 1 (Không tham gia / Tham gia)
    'Hoạt động tình nguyện': np.random.randint(0, 2, n_students),
    'Thể thao': np.random.randint(0, 2, n_students),
    'Mạng xã hội': np.random.randint(0, 2, n_students),
    'Dự án nghiên cứu': np.random.randint(0, 2, n_students),
    'Thời gian học ngoài giờ': np.random.uniform(1, 4, n_students)  # Thời gian học ngoài giờ (từ 1 đến 4 giờ mỗi ngày)
}

df = pd.DataFrame(data)

# Hiển thị một phần dữ liệu
print(df.head())

# Bước 1: Chuẩn hóa dữ liệu để đảm bảo tất cả đặc điểm có trọng số tương tự nhau
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Bước 2: Sử dụng KMeans để phân cụm sinh viên thành 3 nhóm (có thể thử với nhiều k khác nhau)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)

# Lấy kết quả phân cụm
df['Nhóm'] = kmeans.labels_

# Bước 3: Trực quan hóa kết quả phân cụm
# Sử dụng PCA để giảm chiều dữ liệu xuống 2 chiều và trực quan hóa
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# Tạo DataFrame với các thành phần PCA
pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
pca_df['Nhóm'] = df['Nhóm']

# Vẽ biểu đồ phân tán với các nhóm khác nhau
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Nhóm', palette='Set2', data=pca_df, s=100, marker='o')
plt.title("Phân cụm sinh viên dựa trên điểm số và hoạt động ngoại khóa", fontsize=15)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title="Nhóm sinh viên")
plt.show()

# Bước 4: Phân tích các nhóm (bạn có thể thêm mô tả dựa trên kết quả phân cụm)
for i in range(3):
    print(f"\nNhóm {i+1}:")
    print(df[df['Nhóm'] == i].describe())
#     Bước 1: Import các thư viện cần thiết
# python
# Copy
# Edit
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# pandas: Thư viện để xử lý dữ liệu dạng bảng (DataFrame).
# numpy: Thư viện cho các phép toán số học và mảng đa chiều.
# matplotlib.pyplot và seaborn: Dùng để vẽ đồ thị và trực quan hóa dữ liệu.
# KMeans từ sklearn.cluster: Thuật toán phân cụm K-means để nhóm sinh viên.
# StandardScaler từ sklearn.preprocessing: Dùng để chuẩn hóa dữ liệu (mang tất cả các đặc điểm về cùng một phạm vi giá trị).
# PCA từ sklearn.decomposition: Giảm số chiều của dữ liệu để trực quan hóa dễ dàng hơn.
# Bước 2: Giả lập dữ liệu sinh viên
# python
# Copy
# Edit
# np.random.seed(42)
# n_students = 100
# data = {
#     'Điểm học tập': np.random.uniform(5, 10, n_students),
#     'Tham gia CLB': np.random.randint(0, 2, n_students),
#     'Hoạt động tình nguyện': np.random.randint(0, 2, n_students),
#     'Thể thao': np.random.randint(0, 2, n_students),
#     'Mạng xã hội': np.random.randint(0, 2, n_students),
#     'Dự án nghiên cứu': np.random.randint(0, 2, n_students),
#     'Thời gian học ngoài giờ': np.random.uniform(1, 4, n_students)
# }
# df = pd.DataFrame(data)
# print(df.head())
# np.random.seed(42): Đảm bảo các số ngẫu nhiên tạo ra trong chương trình sẽ giống nhau mỗi lần chạy lại mã (tái hiện kết quả).
# n_students = 100: Số lượng sinh viên là 100.
# data: Dữ liệu được tạo ra dưới dạng từ điển (dictionary), mỗi khóa là một đặc điểm của sinh viên và giá trị là mảng ngẫu nhiên tương ứng.
# Điểm học tập: Sinh viên có điểm từ 5 đến 10.
# Các hoạt động ngoại khóa (Tham gia CLB, Hoạt động tình nguyện, Thể thao, Mạng xã hội, Dự án nghiên cứu) là các giá trị nhị phân 0 hoặc 1 (không tham gia hoặc tham gia).
# Thời gian học ngoài giờ: Sinh viên học ngoài giờ từ 1 đến 4 giờ mỗi ngày.
# pd.DataFrame(data): Chuyển từ điển data thành DataFrame để dễ dàng xử lý và phân tích.
# print(df.head()): Hiển thị 5 dòng đầu tiên của dữ liệu để kiểm tra.
# Bước 3: Chuẩn hóa dữ liệu
# python
# Copy
# Edit
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(df)
# StandardScaler(): Khởi tạo bộ chuẩn hóa dữ liệu. Bộ chuẩn hóa này sẽ điều chỉnh dữ liệu sao cho các đặc điểm có trung bình = 0 và độ lệch chuẩn = 1, giúp các thuật toán phân tích hoạt động hiệu quả hơn.
# fit_transform(df): Dùng để tính toán các tham số chuẩn hóa (trung bình và độ lệch chuẩn) và áp dụng chuẩn hóa vào toàn bộ dữ liệu.
# Bước 4: Sử dụng KMeans để phân cụm sinh viên
# python
# Copy
# Edit
# kmeans = KMeans(n_clusters=3, random_state=42)
# kmeans.fit(scaled_data)
# KMeans(n_clusters=3, random_state=42): Khởi tạo thuật toán K-means để phân cụm dữ liệu thành 3 nhóm (n_clusters=3). random_state=42 giúp đảm bảo các kết quả phân cụm là cố định.
# kmeans.fit(scaled_data): Áp dụng thuật toán K-means vào dữ liệu đã được chuẩn hóa để phân cụm sinh viên.
# Bước 5: Lưu kết quả phân cụm vào DataFrame
# python
# Copy
# Edit
# df['Nhóm'] = kmeans.labels_
# kmeans.labels_: Đây là nhãn của các nhóm mà thuật toán K-means gán cho từng sinh viên.
# df['Nhóm'] = kmeans.labels_: Gắn nhãn nhóm vào DataFrame df mới với cột 'Nhóm'.
# Bước 6: Giảm chiều dữ liệu xuống 2 chiều bằng PCA để trực quan hóa
# python
# Copy
# Edit
# pca = PCA(n_components=2)
# pca_components = pca.fit_transform(scaled_data)
# PCA(n_components=2): Khởi tạo PCA với 2 thành phần chính (2 chiều) để giảm số chiều của dữ liệu.
# pca.fit_transform(scaled_data): Áp dụng PCA vào dữ liệu chuẩn hóa và giảm xuống 2 chiều.
# Bước 7: Tạo DataFrame từ kết quả PCA và gán nhãn nhóm
# python
# Copy
# Edit
# pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
# pca_df['Nhóm'] = df['Nhóm']
# pca_components: Đây là kết quả từ PCA, bao gồm các thành phần giảm chiều (PCA1, PCA2).
# pd.DataFrame(pca_components, columns=['PCA1', 'PCA2']): Chuyển kết quả PCA thành DataFrame với tên các cột là 'PCA1' và 'PCA2'.
# pca_df['Nhóm'] = df['Nhóm']: Gắn nhãn nhóm vào DataFrame pca_df.
# Bước 8: Trực quan hóa kết quả phân cụm
# python
# Copy
# Edit
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='PCA1', y='PCA2', hue='Nhóm', palette='Set2', data=pca_df, s=100, marker='o')
# plt.title("Phân cụm sinh viên dựa trên điểm số và hoạt động ngoại khóa", fontsize=15)
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend(title="Nhóm sinh viên")
# plt.show()
# plt.figure(figsize=(8, 6)): Tạo một hình (figure) với kích thước 8x6 inch.
# sns.scatterplot(...): Vẽ biểu đồ phân tán (scatter plot) với các thành phần PCA1 và PCA2 trên trục hoành và tung. Các điểm được phân biệt theo nhóm (hue='Nhóm') với màu sắc khác nhau (palette='Set2').
# plt.title(...): Thêm tiêu đề cho biểu đồ.
# plt.xlabel(...) và plt.ylabel(...): Đặt tên cho các trục X và Y.
# plt.legend(...): Thêm chú thích cho biểu đồ, xác định các nhóm sinh viên.
# plt.show(): Hiển thị biểu đồ.
# Bước 9: Phân tích kết quả phân cụm
# python
# Copy
# Edit
# for i in range(3):
#     print(f"\nNhóm {i+1}:")
#     print(df[df['Nhóm'] == i].describe())
# for i in range(3):: Lặp qua 3 nhóm (từ 0 đến 2).
# df[df['Nhóm'] == i]: Lọc ra các sinh viên thuộc nhóm i.
# describe(): Hiển thị các thống kê mô tả cho nhóm sinh viên đó, bao gồm các giá trị như trung bình, độ lệch chuẩn, min, max, v.v.
