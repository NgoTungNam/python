# ham ngau nhien tao mang 2 chieu 2.3 ,3 chieu 2.3.3 ,4 chieu 2.3.4.5 ,5 chieu 2.2.2.3.4
import numpy as np

# Mảng 2 chiều (2x3)
mang_2d = np.random.rand(2, 3)  # Sinh mảng ngẫu nhiên với 2 hàng và 3 cột
print("Mảng 2 chiều (2x3):")
print(mang_2d)

# Mảng 3 chiều (2x3x3)
mang_3d = np.random.rand(2, 3, 3)  # Sinh mảng ngẫu nhiên với kích thước 2x3x3
print("\nMảng 3 chiều (2x3x3):")
print(mang_3d)

# Mảng 4 chiều (2x3x4x5)
mang_4d = np.random.rand(2, 3, 4, 5)  # Sinh mảng ngẫu nhiên với kích thước 2x3x4x5
print("\nMảng 4 chiều (2x3x4x5):")
print(mang_4d)

# Mảng 5 chiều (2x2x2x3x4)
mang_5d = np.random.rand(2, 2, 2, 3, 4)  # Sinh mảng ngẫu nhiên với kích thước 2x2x2x3x4
print("\nMảng 5 chiều (2x2x2x3x4):")
print(mang_5d)
