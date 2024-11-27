# import pandas as pd
# from sklearn.model_selection import train_test_split

# # 读取CSV文件
# df = pd.read_csv("ICAA20K_csv/annotation.csv")

# # 按8比2随机分割数据
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# # 将分割后的数据保存为新的CSV文件
# train_df.to_csv("ICAA20K_csv/train.csv", index=False)
# test_df.to_csv("ICAA20K_csv/test.csv", index=False)

import pandas as pd

# 读取CSV文件
df = pd.read_csv("ICAA20K_csv/annotation.csv")

# 选择需要计算的列
column_names = ["holistic_color", "temperature", "colorfulness", "color_harmony"]
for column_name in column_names:
    # 计算均值和方差
    mean_value = (df[column_name] / 10).mean()
    variance_value = (df[column_name] / 10).std()

    # 输出结果
    print(f"{column_name}均值: {mean_value}")
    print(f"{column_name}标准差: {variance_value}")

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 读取CSV文件
# df = pd.read_csv("ICAA20K_csv/annotation.csv")

# # 设置图形的尺寸
# plt.figure(figsize=(15, 5))

# # 绘制temperature列的平滑直方图
# plt.subplot(1, 3, 1)
# sns.histplot(df["temperature"], kde=True, color="blue")
# plt.title("Temperature Distribution")

# # 绘制colorfulness列的平滑直方图
# plt.subplot(1, 3, 2)
# sns.histplot(df["colorfulness"], kde=True, color="green")
# plt.title("Colorfulness Distribution")

# # 绘制color_harmony列的平滑直方图
# plt.subplot(1, 3, 3)
# sns.histplot(df["color_harmony"], kde=True, color="red")
# plt.title("Color Harmony Distribution")

# # 显示图形
# plt.tight_layout()
# plt.show()
# plt.savefig("dist.jpg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv("ICAA20K_csv/annotation.csv")

# 设置图形的尺寸
plt.figure(figsize=(15, 5))

# 绘制temperature列的核密度估计图，并进行染色
plt.subplot(1, 3, 1)
sns.kdeplot(df["temperature"], shade=True, color="blue")
plt.title("Temperature Distribution")

# 绘制colorfulness列的核密度估计图，并进行染色
plt.subplot(1, 3, 2)
sns.kdeplot(df["colorfulness"], shade=True, color="green")
plt.title("Colorfulness Distribution")

# 绘制color_harmony列的核密度估计图，并进行染色
plt.subplot(1, 3, 3)
sns.kdeplot(df["color_harmony"], shade=True, color="red")
plt.title("Color Harmony Distribution")

# 显示图形
plt.tight_layout()
plt.show()
plt.savefig("dist.jpg")
