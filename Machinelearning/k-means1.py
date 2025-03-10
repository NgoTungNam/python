# Tao ngau nhien n diem du lieu phan n diem thanh k cum bat ky 
import numpy as np

# Giả sử bạn đã có dữ liệu X (mảng có dạng (n, m) với n điểm và m đặc trưng)
def random_k_means(X, k):
    
    n = X.shape[0]  # Số lượng điểm
    # Chọn ngẫu nhiên k chỉ số điểm từ dữ liệu
    random_indices = np.random.choice(n, size=k, replace=False)
    centroids = X[random_indices]  # Lấy k điểm làm tâm cụm
    
    # Phân chia ngẫu nhiên dữ liệu thành k cụm
    labels = np.random.randint(0, k, size=n)
    
    return labels, centroids
