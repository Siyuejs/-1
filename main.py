import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 配置数据集路径
file_path = 'data/yellow_tripdata_2023-01.parquet'


# ---------------- M1: 数据处理与特征工程 ----------------

def load_and_report(path):
    """加载数据并生成初步质量报告"""
    print("--- 1. 正在加载数据并生成报告 ---")
    df = pd.read_parquet(path)

    # 缺失率统计
    missing_rate = df.isnull().mean() * 100
    print("\n各字段缺失率 (%):")
    print(missing_rate[missing_rate > 0])

    # 异常值统计
    outliers = {
        "距离为负或零": (df['trip_distance'] <= 0).sum(),
        "总金额为负或零": (df['total_amount'] <= 0).sum(),
        "乘客人数为零": (df['passenger_count'] == 0).sum()
    }
    print("\n异常值统计:")
    for k, v in outliers.items():
        print(f"{k}: {v} 条")
    return df


def clean_data(df):
    """数据清洗逻辑及理由说明"""
    df_cleaned = df.copy()

    # 策略 1: 删除行程距离 <= 0 的记录。理由：无效行程，无法反映真实出行需求。
    df_cleaned = df_cleaned[df_cleaned['trip_distance'] > 0]

    # 策略 2: 删除总费用 <= 0 的记录。理由：商业数据中费用异常通常为测试或系统错误。
    df_cleaned = df_cleaned[df_cleaned['total_amount'] > 0]

    # 策略 3: 填充缺失值。理由：乘客人数缺失时设为众数1，保留其余有效字段。
    if df_cleaned['passenger_count'].isnull().any():
        df_cleaned['passenger_count'] = df_cleaned['passenger_count'].fillna(df_cleaned['passenger_count'].mode()[0])

    # 策略 4: 剔除未知区域 ID (264, 265)。理由：这两个ID代表未知地点，会干扰热度分析。
    df_cleaned = df_cleaned[~df_cleaned['PULocationID'].isin([264, 265])]

    print(f"\n--- 2. 清洗完成。保留记录数: {len(df_cleaned)} ---")
    return df_cleaned


def feature_engineering(df):
    """特征提取与衍生特征设计"""
    print("--- 3. 正在提取特征 ---")
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # 基础特征：小时、星期
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek

    # 是否高峰时段 (工作日早晚高峰)
    df['is_rush_hour'] = df.apply(lambda x: 1 if (x['day_of_week'] < 5) and
                                                 (7 <= x['pickup_hour'] <= 9 or 16 <= x['pickup_hour'] <= 19) else 0,
                                  axis=1)

    # 衍生特征 1: 行程时长（分钟）。意义：衡量拥堵情况。
    df['duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df['duration_min'] > 0) & (df['duration_min'] < 180)]

    # 衍生特征 2: 平均每英里费用。意义：反映不同路段或时段的盈利效率。
    df['fare_per_mile'] = df['fare_amount'] / df['trip_distance']

    return df


# ---------------- M2: 分析与可视化 ----------------

# 设置绘图风格和中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_m2_visualizations(df):
    """生成作业要求的 4 项图表"""
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    print("--- 4. 正在生成可视化图表并保存至 outputs/ ---")

    # 1. 出行需求时间规律 (修正了之前的字段名报错)
    plt.figure(figsize=(12, 6))
    df['is_weekend_label'] = df['day_of_week'].apply(lambda x: '周末' if x >= 5 else '工作日')
    # 使用计数统计订单量
    sns.lineplot(data=df, x='pickup_hour', y='fare_amount', hue='is_weekend_label', estimator=len)
    plt.title('工作日 vs 周末：24小时出行需求规律')
    plt.ylabel('订单数量')
    plt.savefig('outputs/m2_hourly_demand.png')
    plt.close()

    # 2. 区域热度分析 (Top 10)
    plt.figure(figsize=(12, 6))
    top_10 = df['PULocationID'].value_counts().head(10)
    sns.barplot(x=top_10.index, y=top_10.values, palette='magma')
    plt.title('订单量最高的 Top 10 上车区域 ID')
    plt.savefig('outputs/m2_top_regions.png')
    plt.close()

    # 3. 车费影响因素 (距离 vs 车费)
    plt.figure(figsize=(10, 6))
    sample_df = df.sample(n=min(2000, len(df)))  # 采样防止绘图卡顿
    sns.scatterplot(data=sample_df, x='trip_distance', y='fare_amount', alpha=0.5)
    plt.title('行程距离与车费金额的分布关系')
    plt.savefig('outputs/m2_fare_vs_distance.png')
    plt.close()

    # 4. 自选洞察：小费比例与付款方式的关系
    plt.figure(figsize=(10, 6))
    df['tip_percent'] = (df['tip_amount'] / df['fare_amount']) * 100
    tip_df = df[(df['tip_percent'] > 0) & (df['tip_percent'] < 50)]
    sns.boxplot(data=tip_df, x='payment_type', y='tip_percent')
    plt.title('不同付款方式的小费比例对比 (1=信用卡, 2=现金)')
    plt.savefig('outputs/m2_tip_analysis.png')
    plt.close()

    print("✅ 所有图表已成功生成！")


# ---------------- 整个项目的执行入口 ----------------

if __name__ == "__main__":
    # 执行 M1
    raw_data = load_and_report(file_path)
    cleaned_data = clean_data(raw_data)
    final_data = feature_engineering(cleaned_data)

    # 执行 M2
    run_m2_visualizations(final_data)

    print("\n[项目提示]: 请在 PyCharm 左侧目录中查看 'outputs' 文件夹获取图表。")