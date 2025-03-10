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
    # Chọn k điểm ngẫu nhiên làm tam_cum ban đầu
    chi_so_ngau_nhien = np.random.choice(len(diem), k, replace=False)
    tam_cum = diem[chi_so_ngau_nhien]
    
    for i in range(toi_da_so_vong_lap):
        # Gán mỗi điểm vào cụm gần nhất
        cum = gan_cac_cum(diem, tam_cum)
        
        # Cập nhật lại các tam_cum
        tam_cum_moi = cap_nhat_tam_cum(diem, cum, k)
        
        # Kiểm tra sự thay đổi của tam_cum
        if np.all(tam_cum == tam_cum_moi):
            break
        tam_cum = tam_cum_moi
    
    return tam_cum, cum

# Ví dụ sử dụng
if __name__ == "__main__":
    # Tạo một tập dữ liệu giả lập
    np.random.seed(42)
    diem = np.vstack((
        np.random.randn(100, 2) + np.array([5, 5]),
        np.random.randn(100, 2) + np.array([10, 10]),
        np.random.randn(100, 2) + np.array([20, 20])
    ))

    k = 3  # Số lượng cụm
    tam_cum, cum = k_means(diem, k)

    # Vẽ đồ thị
    plt.scatter(diem[:, 0], diem[:, 1], c=cum, cmap='viridis')
    plt.scatter(tam_cum[:, 0], tam_cum[:, 1], s=200, c='red', marker='X')  # Vẽ tam_cum
    plt.title('Phân cụm K-means')
    plt.show()

    # Viet chuong trinh tao ngau nhien 100đ du lieu truc quan hoa cac diem du lieu tren mat phang
    # su dung k mean phan thanh k cum truc quan hoa kq phan cum
