import numpy as np
import matplotlib.pyplot as plt

# Hàm tính toán khoảng cách Euclidean giữa hai điểm
def khoang_cach_euclidean(diem1, diem2):
    return np.sqrt(np.sum((diem1 - diem2) ** 2))

# Hàm tìm kiếm chỉ số cụm gần nhất cho mỗi điểm
def gan_cac_cum(diem, tam_cum):
    cum = []
    for d in diem:
        khoang_cach = [khoang_cach_euclidean(d, tc) for tc in tam_cum]
        tam_cum_gan_nhat = np.argmin(khoang_cach)
        cum.append(tam_cum_gan_nhat)
    return cum

# Hàm tính toán trung tâm của mỗi cụm
def cap_nhat_tam_cum(diem, cum, k):
    tam_cum_moi = np.zeros((k, diem.shape[1]))
    for i in range(k):
        diem_trong_cum = diem[np.array(cum) == i]
        tam_cum_moi[i] = np.mean(diem_trong_cum, axis=0)
    return tam_cum_moi

# Thuật toán K-means
def k_means(diem, k, toi_da_so_vong_lap=100):
    chi_so_ngau_nhien = np.random.choice(len(diem), k, replace=False)
    tam_cum = diem[chi_so_ngau_nhien]
    
    for i in range(toi_da_so_vong_lap):
        cum = gan_cac_cum(diem, tam_cum)
        tam_cum_moi = cap_nhat_tam_cum(diem, cum, k)
        
        if np.all(tam_cum == tam_cum_moi):
            break
        tam_cum = tam_cum_moi
    
    return tam_cum, cum

# Tạo 100 điểm dữ liệu ngẫu nhiên
np.random.seed(42)
diem = np.random.randn(100, 2)  # 100 điểm ngẫu nhiên trong không gian 2 chiều

# Số cụm cần phân chia
k = 3

# Sử dụng thuật toán K-means để phân cụm
tam_cum, cum = k_means(diem, k)

# Trực quan hóa kết quả
plt.figure(figsize=(8, 6))

# Vẽ các điểm dữ liệu với màu sắc theo cụm
plt.scatter(diem[:, 0], diem[:, 1], c=cum, cmap='viridis', marker='o', s=100, edgecolor='k')

# Vẽ các trung tâm của các cụm
plt.scatter(tam_cum[:, 0], tam_cum[:, 1], s=300, c='red', marker='X', label='Trung tâm cụm')

# Thêm tiêu đề và nhãn cho đồ thị
plt.title('Phân cụm K-means với 3 cụm')
plt.xlabel('Tọa độ X')
plt.ylabel('Tọa độ Y')
plt.legend()

# Hiển thị đồ thị
plt.show()


