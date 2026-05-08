import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 配置数据集路径
file_path = 'data/yellow_tripdata_2023-01.parquet'


# ---------------- M1: 数据处理与特征工程 ----------------

def load_and_report(path):
    print("--- 1. 正在加载数据并生成报告 ---")
    df = pd.read_parquet(path)
    # 异常值统计逻辑
    outliers = {
        "距离为负或零": (df['trip_distance'] <= 0).sum(),
        "总金额为负或零": (df['total_amount'] <= 0).sum(),
        "乘客人数为零": (df['passenger_count'] == 0).sum()
    }
    print("异常值统计:", outliers)
    return df


def clean_data(df):
    df_cleaned = df.copy()
    df_cleaned = df_cleaned[(df_cleaned['trip_distance'] > 0) & (df_cleaned['total_amount'] > 0)]
    if df_cleaned['passenger_count'].isnull().any():
        df_cleaned['passenger_count'] = df_cleaned['passenger_count'].fillna(df_cleaned['passenger_count'].mode()[0])
    df_cleaned = df_cleaned[~df_cleaned['PULocationID'].isin([264, 265])]
    print(f"--- 2. 清洗完成。保留记录数: {len(df_cleaned)} ---")
    return df_cleaned


def feature_engineering(df):
    print("--- 3. 正在提取特征 ---")
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    # 计算行程时长
    df['duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df['duration_min'] > 0) & (df['duration_min'] < 180)]
    return df


# ---------------- M2: 分析与可视化 ----------------

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_m2_visualizations(df):
    if not os.path.exists('outputs'): os.makedirs('outputs')
    print("--- 4. 正在生成可视化图表 ---")

    # 需求趋势
    plt.figure(figsize=(10, 5))
    df['is_weekend'] = df['day_of_week'].apply(lambda x: '周末' if x >= 5 else '工作日')
    sns.lineplot(data=df, x='pickup_hour', y='fare_amount', hue='is_weekend', estimator=len)
    plt.title('24小时出行需求规律')
    plt.savefig('outputs/m2_hourly_demand.png')
    plt.close()

    # 区域热度
    plt.figure(figsize=(10, 5))
    top_10 = df['PULocationID'].value_counts().head(10)
    sns.barplot(x=top_10.index, y=top_10.values)
    plt.title('Top 10 热门区域')
    plt.savefig('outputs/m2_top_regions.png')
    plt.close()
    print("✅ M2 可视化图表已保存在 outputs/ 文件夹")


# ---------------- M3: 预测模型模块 ----------------

def train_m3_models(df):
    print("\n--- 5. 正在训练 M3 预测模型 ---")
    model_df = df.groupby(['pickup_hour', 'day_of_week', 'PULocationID']).size().reset_index(name='demand')
    X = model_df[['pickup_hour', 'day_of_week', 'PULocationID']]
    y = model_df['demand']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 神经网络
    nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
    nn_model.fit(X_train_s, y_train)

    # 绘制 Loss 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(nn_model.loss_curve_)
    plt.title('神经网络训练 Loss 曲线')
    plt.savefig('outputs/m3_loss_curve.png')
    plt.close()

    print("✅ M3 模型训练完毕并保存 Loss 曲线")
    return nn_model, scaler


# ---------------- M4: 命令行问答系统 ----------------

def run_m4_chatbot(df, model, scaler):
    print("\n" + "=" * 40)
    print("🚕 欢迎进入纽约出租车智能问答系统！")
    print("您可以输入：'哪里最热门'、'8点的需求'、'预测需求'、'平均车费'、'退出'")
    print("=" * 40)

    while True:
        query = input("\n👤 请提问: ").strip()
        if query.lower() in ['退出', 'quit', 'exit']: break

        # 逻辑判断实现 5 种问题类型
        if '热门' in query or '区域' in query:
            top_zone = df['PULocationID'].value_counts().idxmax()
            print(f"🤖 系统结论：最热门的区域ID是 {top_zone}。详情见 outputs/m2_top_regions.png")

        elif '时段' in query or '点' in query:
            print(f"🤖 系统结论：出行规律显示在早晚高峰需求最高。详情见 outputs/m2_hourly_demand.png")

        elif '预测' in query or '需求' in query:
            # 默认预测：周一(0), 早上8点, 100号区域
            test_data = pd.DataFrame([[8, 0, 100]], columns=['pickup_hour', 'day_of_week', 'PULocationID'])
            pred = model.predict(scaler.transform(test_data))[0]
            print(f"🤖 系统预测：周一早8点 100号区域的需求量预计为 {int(pred)} 单。")

        elif '费' in query or '钱' in query:
            avg_fare = df['fare_amount'].mean()
            print(f"🤖 系统结论：本月平均单程车费为 ${avg_fare:.2f}。")

        elif '距离' in query or '远' in query:
            avg_dist = df['trip_distance'].mean()
            print(f"🤖 系统结论：平均行程距离为 {avg_dist:.2f} 英里。")

        else:
            print("🤖 抱歉，我听不太懂，可以换个问法（例如问‘预测需求’）？")


# ---------------- 程序总入口 ----------------

if __name__ == "__main__":
    # 执行 M1
    raw_data = load_and_report(file_path)
    cleaned_data = clean_data(raw_data)
    final_data = feature_engineering(cleaned_data)

    # 执行 M2
    run_m2_visualizations(final_data)

    # 执行 M3 (并获取模型供 M4 使用)
    trained_nn, data_scaler = train_m3_models(final_data)

    # 执行 M4
    run_m4_chatbot(final_data, trained_nn, data_scaler)