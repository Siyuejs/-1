import pandas as pd
import numpy as np

# 建议先从官网下载 yellow_tripdata_2023-01.parquet 并放入项目的 data/ 目录下 [cite: 5, 19]
file_path = 'data/yellow_tripdata_2023-01.parquet'


def load_and_report(path):
    # 使用 pandas 加载 Parquet 格式数据
    df = pd.read_parquet(path)

    # --- 数据质量报告 ---
    print("--- 数据质量报告 ---")

    # 1. 缺失率统计
    missing_rate = df.isnull().mean() * 100
    print("\n各字段缺失率 (%):")
    print(missing_rate[missing_rate > 0])

    # 2. 异常值统计
    # 我们关注核心字段：行程距离、车费金额、乘客人数 [cite: 6]
    outliers = {
        "距离为负或零": (df['trip_distance'] <= 0).sum(),
        "总金额为负或零": (df['total_amount'] <= 0).sum(),
        "乘客人数为零": (df['passenger_count'] == 0).sum(),
        "超长行程 (>100英里)": (df['trip_distance'] > 100).sum()
    }
    print("\n异常值统计:")
    for k, v in outliers.items():
        print(f"{k}: {v} 条")

    return df


df_raw = load_and_report(file_path)


def clean_data(df):
    # 复制数据框以保护原数据
    df_cleaned = df.copy()

    # 策略 1: 删除行程距离 <= 0 的记录
    # 理由: 出租车行程必须产生实际位移，此类数据通常为取消订单或记录错误 。
    df_cleaned = df_cleaned[df_cleaned['trip_distance'] > 0]

    # 策略 2: 删除总费用 (total_amount) <= 0 的记录
    # 理由: 商业运营中费用不可能为负数，此类数据多为系统测试或无效记录 。
    df_cleaned = df_cleaned[df_cleaned['total_amount'] > 0]

    # 策略 3: 处理缺失值
    # 理由: 乘客人数 (passenger_count) 若缺失，填充为众数（通常为 1 人）以保留有效行程 。
    if df_cleaned['passenger_count'].isnull().any():
        df_cleaned['passenger_count'] = df_cleaned['passenger_count'].fillna(df_cleaned['passenger_count'].mode()[0])

    # 策略 4: 剔除不合理的地理位置 ID [cite: 6]
    # 理由: 过滤掉未知的 LocationID (通常为 264 或 265)，确保分析集中在纽约市区 。
    df_cleaned = df_cleaned[~df_cleaned['PULocationID'].isin([264, 265])]
    df_cleaned = df_cleaned[~df_cleaned['DOLocationID'].isin([264, 265])]

    print(f"\n清洗完成。保留记录数: {len(df_cleaned)} (原始: {len(df)})")
    return df_cleaned


df_cleaned = clean_data(df_raw)


def feature_engineering(df):
    # 转换时间格式 [cite: 6]
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # --- 基础特征提取  ---
    # 1. 提取小时 (0-23)
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour

    # 2. 提取星期 (0=周一, 6=周日)
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek

    # 3. 是否高峰时段
    # 定义: 工作日 07:00-09:00 (早高峰) 或 16:00-19:00 (晚高峰)
    df['is_rush_hour'] = df.apply(lambda x: 1 if (x['day_of_week'] < 5) and
                                                 (7 <= x['pickup_hour'] <= 9 or 16 <= x['pickup_hour'] <= 19) else 0,
                                  axis=1)

    # --- 自行设计的衍生特征  ---

    # 特征 1: 行程时长 (Trip_Duration_Minutes)
    # 意义: 结合行程距离可以分析路况拥堵程度，是预测车费的关键因子。
    df['duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    # 过滤掉时长 <= 0 或 > 180 分钟的极端异常值
    df = df[(df['duration_min'] > 0) & (df['duration_min'] < 180)]

    # 特征 2: 订单价值密度 (Value_Density)
    # 意义: 计算每分钟产生的车费 (fare_amount / duration_min)，反映了司机的接单效率。
    df['value_density'] = df['fare_amount'] / df['duration_min']

    print("\n特征提取完成。新增特征: pickup_hour, day_of_week, is_rush_hour, duration_min, value_density")
    return df


df_final = feature_engineering(df_cleaned)