import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 自定义数据加载函数
def load_data(label_path, data_dir):
    label_df = pd.read_csv(label_path)
    features = []
    labels = []
    for idx in range(len(label_df)):
        file_name = label_df.iloc[idx, 0]
        label = label_df.iloc[idx, 1:].values.astype('float32')

        # 读取csv文件中的数据
        file_path = os.path.join(data_dir, str(file_name) + '.csv')
        data = pd.read_csv(file_path, header=None).values.astype('float32').flatten()

        features.append(data)
        labels.append(label)

    return features, labels

# 加载训练数据
train_label = './redLowdata/fold_1/train_labels.csv'
train_data_dir = './redLowdata/fold_1/train_data'
X_train, y_train = load_data(train_label, train_data_dir)

# 加载验证数据
eval_label = './redLowdata/fold_1/val_labels.csv'
eval_data_dir = './redLowdata/fold_1/val_data'
X_eval, y_eval = load_data(eval_label, eval_data_dir)

# 初始化并训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_eval)

# 评估模型
mse = mean_squared_error(y_eval, y_pred)
print(f'Mean Squared Error: {mse}')