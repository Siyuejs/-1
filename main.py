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
    missing_rate = df.isnull().mean() * 100
    print("\n各字段缺失率 (%):")
    print(missing_rate[missing_rate > 0])
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
    df_cleaned = df.copy()
    df_cleaned = df_cleaned[df_cleaned['trip_distance'] > 0]
    df_cleaned = df_cleaned[df_cleaned['total_amount'] > 0]
    if df_cleaned['passenger_count'].isnull().any():
        df_cleaned['passenger_count'] = df_cleaned['passenger_count'].fillna(df_cleaned['passenger_count'].mode()[0])
    df_cleaned = df_cleaned[~df_cleaned['PULocationID'].isin([264, 265])]
    print(f"\n--- 2. 清洗完成。保留记录数: {len(df_cleaned)} ---")
    return df_cleaned

def feature_engineering(df):
    print("--- 3. 正在提取特征 ---")
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['is_rush_hour'] = df.apply(lambda x: 1 if (x['day_of_week'] < 5) and
                                                 (7 <= x['pickup_hour'] <= 9 or 16 <= x['pickup_hour'] <= 19) else 0,
                                  axis=1)
    df['duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df['duration_min'] > 0) & (df['duration_min'] < 180)]
    df['fare_per_mile'] = df['fare_amount'] / df['trip_distance']
    return df

# ---------------- M2: 分析与可视化 ----------------

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_m2_visualizations(df):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    print("--- 4. 正在生成可视化图表并保存至 outputs/ ---")

    # 1. 出行需求时间规律
    plt.figure(figsize=(12, 6))
    df['is_weekend_label'] = df['day_of_week'].apply(lambda x: '周末' if x >= 5 else '工作日')
    sns.lineplot(data=df, x='pickup_hour', y='fare_amount', hue='is_weekend_label', estimator=len)
    plt.title('工作日 vs 周末：24小时出行需求规律')
    plt.ylabel('订单数量')
    plt.savefig('outputs/m2_hourly_demand.png')
    plt.close()

    # 2. 区域热度分析
    plt.figure(figsize=(12, 6))
    top_10 = df['PULocationID'].value_counts().head(10)
    sns.barplot(x=top_10.index, y=top_10.values, palette='magma')
    plt.title('订单量最高的 Top 10 上车区域 ID')
    plt.savefig('outputs/m2_top_regions.png')
    plt.close()

    # 3. 车费影响因素
    plt.figure(figsize=(10, 6))
    sample_df = df.sample(n=min(2000, len(df)))
    sns.scatterplot(data=sample_df, x='trip_distance', y='fare_amount', alpha=0.5)
    plt.title('行程距离与车费金额的关系')
    plt.savefig('outputs/m2_fare_vs_distance.png')
    plt.close()

    # 4. 自选洞察：小费分析
    plt.figure(figsize=(10, 6))
    df['tip_percent'] = (df['tip_amount'] / df['fare_amount']) * 100
    tip_df = df[(df['tip_percent'] > 0) & (df['tip_percent'] < 50)]
    sns.boxplot(data=tip_df, x='payment_type', y='tip_percent')
    plt.title('不同付款方式的小费比例对比')
    plt.savefig('outputs/m2_tip_analysis.png')
    plt.close()
    print("✅ M2 图表已成功生成！")

# ---------------- M3: 预测模型模块 ----------------

def run_m3_models(df):
    print("\n--- 5. 正在启动 M3 预测模型模块 ---")
    model_df = df.groupby(['pickup_hour', 'day_of_week', 'PULocationID']).size().reset_index(name='demand')
    X = model_df[['pickup_hour', 'day_of_week', 'PULocationID']]
    y = model_df['demand']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("正在训练随机森林模型...")
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    print("正在训练神经网络模型 (此步骤较慢，请稍候)...")
    nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    nn_model.fit(X_train_scaled, y_train)
    nn_preds = nn_model.predict(X_test_scaled)

    def get_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return mae, rmse

    rf_mae, rf_rmse = get_metrics(y_test, rf_preds)
    nn_mae, nn_rmse = get_metrics(y_test, nn_preds)

    print("\n" + "=" * 35)
    print(f"随机森林报告: MAE = {rf_mae:.2f}, RMSE = {rf_rmse:.2f}")
    print(f"神经网络报告: MAE = {nn_mae:.2f}, RMSE = {nn_rmse:.2f}")
    print("=" * 35)

    plt.figure(figsize=(8, 4))
    plt.plot(nn_model.loss_curve_)
    plt.title('神经网络训练 Loss 曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('outputs/m3_loss_curve.png')
    plt.close()
    print("✅ M3 模型对比完成，Loss 曲线已保存。")

# ---------------- 整个项目的唯一执行入口 ----------------

if __name__ == "__main__":
    # 1. 运行 M1
    raw_data = load_and_report(file_path)
    cleaned_data = clean_data(raw_data)
    final_data = feature_engineering(cleaned_data)

    # 2. 运行 M2
    run_m2_visualizations(final_data)

    # 3. 运行 M3
    run_m3_models(final_data)

    print("\n🎉 所有任务全部执行完毕！请检查 outputs 文件夹。")