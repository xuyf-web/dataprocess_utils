#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-

# 根据用户指定的经纬度范围筛选站点
# Filter stations based on user-specified latitude and longitude range

import pandas as pd
import numpy as np
import os
import glob
import logging
import time
from pathlib import Path

# 获取已有的logger
logger = logging.getLogger(__name__)

# 输入参数
STATION_FILE = "/data8/xuyf/data/obs/meteo_ncdc/doc/China_Stations_with_decimal_coords.xlsx"
DATA_DIR = "/data8/xuyf/data/obs/meteo_ncdc"

# 经纬度范围 (示例值，可根据需要修改)
MIN_LAT = 30.0
MAX_LAT = 32.0  # 缩小范围以减少处理时间
MIN_LON = 115.0
MAX_LON = 117.0  # 缩小范围以减少处理时间

# 经纬度容差值 (单位：度)
COORD_TOLERANCE = 0.5  # 约55公里

# 年份限制 (可选，用于限制CSV文件搜索范围)
# 如果设置为None，则搜索所有年份的数据
YEAR_LIMIT = None  # 设置为None表示不限制年份

def filter_stations_by_coords(df, min_lat, max_lat, min_lon, max_lon):
    """
    根据经纬度范围筛选站点
    """
    mask = (
        (df['纬度(十进制度)'] >= min_lat) & 
        (df['纬度(十进制度)'] <= max_lat) & 
        (df['经度(十进制度)'] >= min_lon) & 
        (df['经度(十进制度)'] <= max_lon)
    )
    return df[mask]

def find_csv_files_for_stations(station_ids, data_dir, year_limit=None):
    """
    查找站点对应的CSV文件
    站点号格式转换: 例如58015 -> 580150xxxxx.csv
    
    Parameters:
    -----------
    station_ids : list
        站点ID列表
    data_dir : str
        数据目录
    year_limit : str, optional
        限制年份，格式为'YYYY'
        
    Returns:
    --------
    csv_files : dict
        站点ID和对应CSV文件的字典
    stations_with_data : list
        有数据的站点ID列表
    """
    csv_files = {}
    stations_with_data = []
    
    # 如果指定了年份，首先检查对应年份目录是否存在
    if year_limit:
        year_dir = os.path.join(data_dir, year_limit)
        if os.path.exists(year_dir) and os.path.isdir(year_dir):
            logger.info(f"将搜索限制在 {year_limit} 年份目录")
            search_dir = year_dir
        else:
            logger.warning(f"年份目录 {year_limit} 不存在，将搜索整个数据目录")
            search_dir = data_dir
    else:
        search_dir = data_dir
    
    start_time = time.time()
    
    for i, station_id in enumerate(station_ids):
        # 站点号转换为CSV文件名前缀 (区站号 + '0')
        file_prefix = f"{station_id}0"
        
        # 查找匹配的CSV文件
        pattern = os.path.join(search_dir, "**", f"{file_prefix}*.csv")
        matched_files = glob.glob(pattern, recursive=True)
        
        if matched_files:
            csv_files[station_id] = matched_files
            stations_with_data.append(station_id)
            logger.info(f"站点 {station_id} 找到 {len(matched_files)} 个CSV文件")
        else:
            logger.warning(f"站点 {station_id} 未找到对应的CSV文件")
        
        # 每处理10个站点显示进度
        if (i + 1) % 10 == 0 or i == len(station_ids) - 1:
            elapsed = time.time() - start_time
            logger.info(f"已处理 {i+1}/{len(station_ids)} 个站点 (耗时: {elapsed:.2f}秒)")
    
    return csv_files, stations_with_data

def compare_station_coords(station_df, csv_file, tolerance=0.1):
    """
    比较站点表中的经纬度和CSV文件中的经纬度是否一致
    
    Parameters:
    -----------
    station_df : DataFrame
        站点信息表
    csv_file : str
        CSV文件路径
    tolerance : float, optional
        容差值，单位为度
        
    Returns:
    --------
    bool
        True表示经纬度一致（在容差范围内），False表示不一致
    """
    try:
        # 读取CSV文件的第一行获取经纬度
        csv_df = pd.read_csv(csv_file, nrows=1)
        
        if 'LATITUDE' not in csv_df.columns or 'LONGITUDE' not in csv_df.columns:
            logger.warning(f"CSV文件 {os.path.basename(csv_file)} 中没有经纬度列")
            return False
        
        csv_lat = csv_df['LATITUDE'].iloc[0]
        csv_lon = csv_df['LONGITUDE'].iloc[0]
        
        # 读取CSV文件中的NAME列
        csv_name = "未知"
        if 'NAME' in csv_df.columns:
            csv_name = csv_df['NAME'].iloc[0]
        
        # 从站点表中获取经纬度
        station_id = os.path.basename(csv_file)[:5]  # 从文件名提取站点ID
        station_row = station_df[station_df['区站号'] == int(station_id)]
        
        if station_row.empty:
            logger.warning(f"站点表中没有ID为 {station_id} 的站点")
            return False
        
        station_lat = station_row['纬度(十进制度)'].iloc[0]
        station_lon = station_row['经度(十进制度)'].iloc[0]
        station_name = station_row['站名'].iloc[0]
        
        # 计算差异
        lat_diff = abs(csv_lat - station_lat)
        lon_diff = abs(csv_lon - station_lon)
        
        if lat_diff > tolerance or lon_diff > tolerance:
            logger.warning(
                f"站点 {station_id} ({station_name}) 经纬度不一致: "
                f"站点表: {station_lat:.4f}, {station_lon:.4f}, "
                f"CSV: {csv_lat:.4f}, {csv_lon:.4f}, "
                f"差异: {lat_diff:.4f}, {lon_diff:.4f}, "
                f"CSV中站名: {csv_name}"
            )
            return False
        else:
            logger.info(
                f"站点 {station_id} ({station_name}) 经纬度一致: "
                f"站点表: {station_lat:.4f}, {station_lon:.4f}, "
                f"CSV: {csv_lat:.4f}, {csv_lon:.4f}, "
                f"CSV中站名: {csv_name}"
            )
            return True
    
    except Exception as e:
        logger.error(f"比较站点 {os.path.basename(csv_file)} 经纬度时出错: {e}")
        return False

# 如果作为独立脚本运行，则配置自己的日志
if __name__ == "__main__":
    # 仅当作为独立脚本运行时才配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("station_filter.log"),
            logging.StreamHandler()
        ]
    )
    
    logger.info("开始筛选站点")
    logger.info(f"经纬度范围: {MIN_LAT}-{MAX_LAT}°N, {MIN_LON}-{MAX_LON}°E")
    logger.info(f"经纬度容差: {COORD_TOLERANCE}°")
    if YEAR_LIMIT:
        logger.info(f"限制年份: {YEAR_LIMIT}")
    
    start_time = time.time()
    
    try:
        # 读取站点文件
        logger.info(f"读取站点文件: {STATION_FILE}")
        station_df = pd.read_excel(STATION_FILE)
        
        # 筛选站点
        filtered_df = filter_stations_by_coords(station_df, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)
        logger.info(f"在指定范围内找到 {len(filtered_df)} 个站点")
        
        # 打印筛选的站点信息
        for _, row in filtered_df.iterrows():
            logger.info(
                f"站点ID: {row['区站号']}, 站名: {row['站名']}, "
                f"经纬度: {row['纬度(十进制度)']:.4f}°N, {row['经度(十进制度)']:.4f}°E"
            )
        
        # 获取站点ID列表
        station_ids = filtered_df['区站号'].astype(str).tolist()
        
        # 查找站点对应的CSV文件
        logger.info("查找站点对应的CSV文件...")
        csv_files, stations_with_data = find_csv_files_for_stations(station_ids, DATA_DIR, YEAR_LIMIT)
        
        logger.info(f"共有 {len(stations_with_data)}/{len(station_ids)} 个站点找到对应的CSV文件")
        
        # 比较站点经纬度
        logger.info("比较站点经纬度信息...")
        consistent_coords = 0
        inconsistent_coords = 0
        
        for station_id, files in csv_files.items():
            # 只检查第一个文件
            if files:
                result = compare_station_coords(station_df, files[0], COORD_TOLERANCE)
                if result:
                    consistent_coords += 1
                else:
                    inconsistent_coords += 1
        
        # 生成结果文件
        logger.info("生成筛选结果文件...")
        result_df = filtered_df[filtered_df['区站号'].astype(str).isin(stations_with_data)]
        result_file = "filtered_stations.xlsx"
        result_df.to_excel(result_file, index=False)
        logger.info(f"结果已保存到: {result_file}")
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 打印结果摘要
        logger.info("========== 筛选结果摘要 ==========")
        logger.info(f"指定范围内站点总数: {len(filtered_df)}")
        logger.info(f"找到CSV数据的站点数: {len(stations_with_data)}")
        logger.info(f"未找到CSV数据的站点数: {len(filtered_df) - len(stations_with_data)}")
        logger.info(f"经纬度一致的站点数: {consistent_coords}")
        logger.info(f"经纬度不一致的站点数: {inconsistent_coords}")
        logger.info(f"总耗时: {total_time:.2f}秒")
        logger.info("================================")
    
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        raise 