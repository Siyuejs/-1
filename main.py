import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1. 全局配置 ---
FILE_PATH = 'data/yellow_tripdata_2023-01.parquet'
# 提示：在此处填入你的 DeepSeek API Key
API_KEY = "你的_DEEPSEEK_API_KEY"
API_URL = "https://api.deepseek.com/chat/completions"


# ---------------- M1: 数据处理与特征工程 ----------------

def load_and_clean_data(path):
    print("--- 1. 正在加载并清洗数据 ---")
    df = pd.read_parquet(path)

    # 清洗：剔除无效距离、零金额和未知区域 (264, 265)
    df = df[(df['trip_distance'] > 0) & (df['total_amount'] > 0)]
    df = df[~df['PULocationID'].isin([264, 265])]

    # 填充乘客人数缺失值
    if df['passenger_count'].isnull().any():
        df['passenger_count'] = df['passenger_count'].fillna(df['passenger_count'].mode()[0])

    # 特征提取
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek

    print(f"✅ 数据处理完成。保留记录数: {len(df)}")
    return df


# ---------------- M2: 分析与可视化 ----------------

def run_m2_visualizations(df):
    if not os.path.exists('outputs'): os.makedirs('outputs')
    print("--- 2. 正在生成可视化图表 ---")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 24小时需求规律
    plt.figure(figsize=(10, 5))
    df['is_weekend'] = df['day_of_week'].apply(lambda x: '周末' if x >= 5 else '工作日')
    sns.lineplot(data=df, x='pickup_hour', y='fare_amount', hue='is_weekend', estimator=len)
    plt.title('24小时出行需求规律')
    plt.savefig('outputs/m2_hourly_demand.png')
    plt.close()

    # 2. 热门上车区域
    plt.figure(figsize=(10, 5))
    top_10 = df['PULocationID'].value_counts().head(10)
    sns.barplot(x=top_10.index, y=top_10.values)
    plt.title('Top 10 热门区域')
    plt.savefig('outputs/m2_top_regions.png')
    plt.close()
    print("✅ 可视化图表已保存至 outputs/ 文件夹")


# ---------------- M3: 预测模型模块 ----------------

def train_m3_models(df):
    print("\n--- 3. 正在训练神经网络模型 ---")
    # 数据聚合
    model_df = df.groupby(['pickup_hour', 'day_of_week', 'PULocationID']).size().reset_index(name='demand')
    X = model_df[['pickup_hour', 'day_of_week', 'PULocationID']]
    y = model_df['demand']

    # 划分与标准化
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 训练 MLP 神经网络 (解决 TensorFlow 环境报错问题)
    nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
    nn_model.fit(X_train_s, y_train)

    # 绘制 Loss 曲线 (作业硬性要求)
    plt.figure(figsize=(8, 4))
    plt.plot(nn_model.loss_curve_)
    plt.title('神经网络训练 Loss 曲线')
    plt.savefig('outputs/m3_loss_curve.png')
    plt.close()

    print("✅ 预测模型训练完毕并保存 Loss 曲线")
    return nn_model, scaler


# ---------------- M4: 智能问答系统 (含 LLM 接入) ----------------

def call_llm_api(query):
    """DeepSeek API 兜底回复逻辑"""
    system_prompt = (
        "你是一个专业的纽约出租车数据分析助手。你已经完成了2023年1月数据的特征工程和神经网络预测。"
        "1. 如果用户的问题不在硬编码规则内，请基于专业知识给出解释。"
        "2. 礼貌拒绝无关话题，并引导用户询问'需求预测'、'热门区域'或'费用规律'。"
    )

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=10)
        return response.json()['choices'][0]['message']['content']
    except:
        return "🤖 [AI 暂时离线] 我目前只能处理关于数据预测、区域排行和费用的固定查询。"


def run_m4_chatbot(df, model, scaler):
    print("\n" + "★" * 20 + " 纽约出租车智能问答系统 V5.0 " + "★" * 20)
    print("支持：1.热门查询 2.时段规律 3.需求预测 4.费用分析 5.AI 智能解释")

    while True:
        query = input("\n👤 请提问 (或输入'quit'退出) > ").strip().lower()
        if query in ['quit', 'exit', '退出']: break

        # 规则引擎逻辑
        if any(w in query for w in ['热门', '区域', '哪里']):
            top_zone = df['PULocationID'].value_counts().idxmax()
            print(f"🤖 系统结论：最热门上车区域 ID 为 {top_zone}。相关图表见 outputs/m2_top_regions.png")

        elif '预测' in query or '需求' in query:
            test_data = pd.DataFrame([[8, 0, 132]], columns=['pickup_hour', 'day_of_week', 'PULocationID'])
            pred = model.predict(scaler.transform(test_data))[0]
            print(f"🤖 系统预测：预计该条件下需求量约为 {int(pred)} 单。")

        elif any(w in query for w in ['费', '钱', '贵']):
            avg_fare = df['fare_amount'].mean()
            print(f"🤖 数据统计：本月平均每程车费为 ${avg_fare:.2f}。")

        # 大模型兜底
        else:
            print("🤖 正在为您询问 AI 专家...")
            print(f"🤖 AI 回复：{call_llm_api(query)}")


# ---------------- 程序主入口 ----------------

if __name__ == "__main__":
    # 执行 M1
    data = load_and_clean_data(FILE_PATH)

    # 执行 M2
    run_m2_visualizations(data)

    # 执行 M3
    nn_model, data_scaler = train_m3_models(data)

    # 执行 M4 (交互式问答)
    run_m4_chatbot(data, nn_model, data_scaler)