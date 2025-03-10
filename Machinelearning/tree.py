import pandas as pd
import math
from collections import Counter

# 1. Hàm load dữ liệu
def load_data(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xls/.xlsx files.")
    return data

# 2. Hàm tính entropy
def calculate_entropy(data, target_column):
    entropy = 0
    total_rows = len(data)
    target_counts = Counter(data[target_column])
    
    for count in target_counts.values():
        probability = count / total_rows
        entropy -= probability * math.log2(probability)
    
    return entropy

# 3. Hàm tính information gain
def calculate_information_gain(data, attribute, target_column):
    total_entropy = calculate_entropy(data, target_column)
    total_rows = len(data)
    attribute_counts = Counter(data[attribute])
    weighted_entropy = 0
    
    for value, count in attribute_counts.items():
        subset = data[data[attribute] == value]
        subset_entropy = calculate_entropy(subset, target_column)
        weighted_entropy += (count / total_rows) * subset_entropy
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

# 4. Hàm xây dựng cây quyết định
def id3(data, original_data, features, target_column, parent_node_class=None):
    # Trường hợp dừng
    if len(Counter(data[target_column])) <= 1:
        return list(data[target_column])[0]
    if len(data) == 0:
        return Counter(original_data[target_column]).most_common(1)[0][0]
    if len(features) == 0:
        return parent_node_class
    
    # Chọn thuộc tính tốt nhất
    gains = {feature: calculate_information_gain(data, feature, target_column) for feature in features}
    best_feature = max(gains, key=gains.get)
    
    # Xây dựng cây
    tree = {best_feature: {}}
    features = [f for f in features if f != best_feature]
    
    for value in set(data[best_feature]):
        subset = data[data[best_feature] == value]
        subtree = id3(subset, original_data, features, target_column, Counter(data[target_column]).most_common(1)[0][0])
        tree[best_feature][value] = subtree
    
    return tree

# 5. Hàm dự đoán
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    value = sample.get(attribute)
    if value in tree[attribute]:
        return predict(tree[attribute][value], sample)
    else:
        return None

# Ví dụ sử dụng
if __name__ == "__main__":
    # Load dữ liệu
    data = load_data('data.csv')  # Đảm bảo file data.csv tồn tại trong cùng thư mục
    target_column = 'Play'  # Thay bằng tên cột mục tiêu của bạn
    features = list(data.columns)
    features.remove(target_column)
    
    # Xây dựng cây quyết định
    decision_tree = id3(data, data, features, target_column)
    print("Decision Tree:", decision_tree)
    
    # Dự đoán
    sample = {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'}
    prediction = predict(decision_tree, sample)
    print("Prediction:", prediction)