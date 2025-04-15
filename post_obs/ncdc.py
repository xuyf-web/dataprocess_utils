#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-

# ============================================================
# 参数配置区域（根据需要修改以下参数）
# ============================================================

# 输入/输出参数
DATA_DIR = '/data8/xuyf/data/obs/meteo_ncdc/2024'  # NCDC数据所在目录
OUTPUT_FILE = '/data8/xuyf/data/obs/meteo_ncdc/output/xishan.xlsx'    # 输出Excel文件路径

# 时间范围参数
START_DATE = '2024-07-22'        # 起始日期，格式：'YYYY-MM-DD'
END_DATE = '2024-09-08'          # 结束日期，格式：'YYYY-MM-DD'

# 空间范围选择参数
# 当同时设置了经纬度范围和站点列表时，此参数决定使用哪种方式过滤数据
# 可选值：'coords'（使用经纬度范围）或 'stations'（使用站点列表）
FILTER_MODE = 'coords'

# 经纬度范围参数
MIN_LAT = 27.0                   # 最小纬度
MAX_LAT = 35.0                   # 最大纬度
MIN_LON = 115.0                  # 最小经度
MAX_LON = 123.0                  # 最大经度

# 站点列表参数
STATION_IDS = None               # 站点ID列表，例如：['58112', '58306']
STATION_NAMES = None             # 站点名称列表，例如：['天柱山', '金寨']

# 数据处理参数
CLEAN_DATA = True                # 是否进行数据清洗（True/False）

# 导入站点筛选模块的功能
STATION_FILE = "/data8/xuyf/data/obs/meteo_ncdc/doc/China_Stations_with_decimal_coords.xlsx"
COORD_TOLERANCE = 0.5            # 经纬度容差值 (单位：度)

# ============================================================
# 以下是脚本代码，一般不需要修改
# ============================================================

import os
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import re
import openpyxl
import glob
import logging
import time
import math
from filter_stations import filter_stations_by_coords, find_csv_files_for_stations, compare_station_coords

# 日志文件路径
LOG_FILE = "extract.log"

# 删除已有的日志文件
if os.path.exists(LOG_FILE):
    try:
        os.remove(LOG_FILE)
        print(f"已删除旧的日志文件: {LOG_FILE}")
    except Exception as e:
        print(f"删除旧的日志文件时出错: {e}")

# 设置日志
# 创建日志格式器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 设置文件处理器，记录所有日志
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# 设置控制台处理器，只记录重要日志
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# 配置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# 移除默认处理器
for hdlr in logger.handlers[:]:
    logger.removeHandler(hdlr)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 设置控制台日志过滤器
class ConsoleFilter(logging.Filter):
    def __init__(self, skip_patterns=None):
        super().__init__()
        self.skip_patterns = skip_patterns or []
    
    def filter(self, record):
        # 检查是否包含要跳过的模式
        for pattern in self.skip_patterns:
            if pattern in record.getMessage():
                return False
        return True

# 设置跳过的日志信息模式
skip_patterns = [
    "未找到对应的CSV文件",
    "文件中的站点ID", 
    "按日期范围过滤",
    "过滤前记录数",
    "过滤后记录数",
    "获取到站点",
    "清洗前变量",
    "清洗后变量",
    "处理气温数据",
    "处理露点温度数据",
    "处理气压数据",
    "处理风向数据",
    "处理风速数据",
    "计算相对湿度",
    "已处理",
    "耗时"
]

# 添加过滤器到控制台处理器
console_filter = ConsoleFilter(skip_patterns)
console_handler.addFilter(console_filter)

# 修改文件处理器的日志级别，保留所有详细日志
file_handler.setLevel(logging.DEBUG)

# 修改控制台处理器的日志级别，只显示重要信息
console_handler.setLevel(logging.INFO)

# 修改控制台过滤器，让重要信息能够显示在屏幕上
# 移除之前的过滤器
console_handler.removeFilter(console_filter)

# 创建新的过滤器，只过滤一些非常详细的调试信息
detailed_debug_patterns = [
    "清洗前变量",
    "从文件获取站点信息",
    "处理文件:",
    "文件中的站点ID",
    "按日期范围过滤",
    "过滤前记录数",
    "过滤后记录数"
]

class InfoPassFilter(logging.Filter):
    def __init__(self, detailed_patterns=None):
        super().__init__()
        self.detailed_patterns = detailed_patterns or []
    
    def filter(self, record):
        # 始终显示 INFO 级别及以上的消息
        if record.levelno >= logging.INFO:
            return True
            
        # 只过滤包含详细调试信息的 DEBUG 消息
        for pattern in self.detailed_patterns:
            if pattern in record.getMessage():
                return False
        
        # 其他 DEBUG 消息通过
        return True

# 添加新过滤器到控制台处理器
console_info_filter = InfoPassFilter(detailed_debug_patterns)
console_handler.addFilter(console_info_filter)

class NCDCExtractor:
    """
    NCDC气象观测数据提取器
    用于按照时间范围和空间范围提取NCDC气象数据并保存为Excel文件
    """
    
    def __init__(self, data_dir, output_file):
        """
        初始化提取器
        
        参数:
            data_dir: NCDC气象数据所在目录
            output_file: 输出Excel文件路径
        """
        self.data_dir = Path(data_dir)
        self.output_file = output_file
        self.station_info = None
        self.filtered_stations = None
        
    def _load_station_info(self):
        """加载站点信息"""
        try:
            # 尝试从China_Stations_with_decimal_coords.xlsx加载站点信息
            station_file = Path(STATION_FILE)
            if station_file.exists():
                self.station_info = pd.read_excel(station_file)
                logger.info(f"成功加载了 {len(self.station_info)} 个站点信息")
                
                # 确保区站号是字符串类型
                if '区站号' in self.station_info.columns:
                    self.station_info['区站号'] = self.station_info['区站号'].astype(str)
            else:
                logger.warning(f"警告: 找不到站点信息文件 {station_file}")
                self.station_info = pd.DataFrame()
        except Exception as e:
            logger.error(f"加载站点信息时出错: {e}")
            self.station_info = pd.DataFrame()
            
    def _get_station_from_name(self, station_names):
        """根据站点名称获取站点ID"""
        if self.station_info is None:
            self._load_station_info()
            
        if self.station_info.empty:
            logger.warning("站点信息为空，无法根据名称查找站点ID")
            return []
            
        # 检查站点信息表中是否有"站名"列
        if '站名' not in self.station_info.columns:
            logger.warning("站点信息表中没有'站名'列，无法根据名称查找站点ID")
            return []
            
        # 根据站点名称查找站点ID
        station_ids = []
        for name in station_names:
            matches = self.station_info[self.station_info['站名'] == name]
            if not matches.empty:
                station_id = str(matches['区站号'].iloc[0])
                station_ids.append(station_id)
                logger.info(f"站点名称 '{name}' 匹配到站点ID: {station_id}")
            else:
                logger.warning(f"找不到名为 '{name}' 的站点")
                
        return station_ids
    
    def _parse_csv_data(self, file_path):
        """解析CSV文件数据"""
        try:
            # 读取CSV文件，设置low_memory=False避免DtypeWarning
            df = pd.read_csv(file_path, low_memory=False)
            
            # 确保DATE列是日期类型
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'])
            
            return df
        except Exception as e:
            logger.error(f"解析文件 {file_path} 时出错: {e}")
            return pd.DataFrame()
    
    def _clean_data(self, df):
        """清洗数据，处理缺失值和单位转换"""
        if df.empty:
            return df
        
        # 复制DataFrame以避免修改原始数据
        cleaned = df.copy()
        
        # 提取所需的气象要素
        try:
            # 1. 处理气温数据 (TMP列)
            if 'TMP' in cleaned.columns:
                # 假设TMP格式如 "+0123,1" 表示 12.3°C
                try:
                    # 提取数值部分
                    cleaned['TEMPERATURE'] = cleaned['TMP'].astype(str).str.extract(r'([+-]\d+)')[0]
                    # 转换为数值类型
                    cleaned['TEMPERATURE'] = pd.to_numeric(cleaned['TEMPERATURE'], errors='coerce')
                    # 除以10得到摄氏度
                    cleaned['TEMPERATURE'] = cleaned['TEMPERATURE'] / 10
                    # 处理缺失值
                    cleaned.loc[cleaned['TEMPERATURE'] >= 999.9, 'TEMPERATURE'] = np.nan
                    # 打印调试信息
                    logger.debug(f"处理气温数据: 提取了 {cleaned['TEMPERATURE'].count()} 个有效值")
                except Exception as e:
                    logger.error(f"处理气温数据时出错: {e}")
            else:
                logger.warning("数据中缺少TMP列，无法处理气温数据")
            
            # 2. 处理露点温度 (DEW列)
            if 'DEW' in cleaned.columns:
                try:
                    # 提取数值部分
                    cleaned['DEWPOINT'] = cleaned['DEW'].astype(str).str.extract(r'([+-]\d+)')[0]
                    # 转换为数值类型
                    cleaned['DEWPOINT'] = pd.to_numeric(cleaned['DEWPOINT'], errors='coerce')
                    # 除以10得到摄氏度
                    cleaned['DEWPOINT'] = cleaned['DEWPOINT'] / 10
                    # 处理缺失值
                    cleaned.loc[cleaned['DEWPOINT'] >= 999.9, 'DEWPOINT'] = np.nan
                    # 打印调试信息
                    logger.debug(f"处理露点温度数据: 提取了 {cleaned['DEWPOINT'].count()} 个有效值")
                except Exception as e:
                    logger.error(f"处理露点温度数据时出错: {e}")
            else:
                logger.warning("数据中缺少DEW列，无法处理露点温度数据")
            
            # 3. 处理海平面气压 (SLP列)
            if 'SLP' in cleaned.columns:
                try:
                    # 提取数值部分
                    cleaned['PRESSURE'] = cleaned['SLP'].astype(str).str.extract(r'(\d+)')[0]
                    # 转换为数值类型
                    cleaned['PRESSURE'] = pd.to_numeric(cleaned['PRESSURE'], errors='coerce')
                    # 除以10得到百帕
                    cleaned['PRESSURE'] = cleaned['PRESSURE'] / 10
                    # 处理缺失值
                    cleaned.loc[cleaned['PRESSURE'] >= 9999.9, 'PRESSURE'] = np.nan
                    # 打印调试信息
                    logger.debug(f"处理气压数据: 提取了 {cleaned['PRESSURE'].count()} 个有效值")
                except Exception as e:
                    logger.error(f"处理气压数据时出错: {e}")
            else:
                logger.warning("数据中缺少SLP列，无法处理气压数据")
            
            # 4. 处理风速数据 (WND列)
            if 'WND' in cleaned.columns:
                try:
                    # 假设WND格式如 "140,1,N,0150,1" 表示风向140度，风速15.0米/秒
                    # 分割WND字段
                    wnd_parts = cleaned['WND'].astype(str).str.split(',', expand=True)
                    
                    if len(wnd_parts.columns) >= 5:
                        # 风向 (第一个值)
                        cleaned['WIND_DIRECTION'] = pd.to_numeric(wnd_parts[0], errors='coerce')
                        # 处理缺失值 (999表示缺失)
                        cleaned.loc[cleaned['WIND_DIRECTION'] >= 999, 'WIND_DIRECTION'] = np.nan
                        
                        # 风速 (第四个值)
                        cleaned['WIND_SPEED'] = pd.to_numeric(wnd_parts[3], errors='coerce')
                        # 除以10得到米/秒
                        cleaned['WIND_SPEED'] = cleaned['WIND_SPEED'] / 10
                        # 处理缺失值 (9999表示缺失)
                        cleaned.loc[cleaned['WIND_SPEED'] >= 999.9, 'WIND_SPEED'] = np.nan
                        
                        # 打印调试信息
                        logger.debug(f"处理风向数据: 提取了 {cleaned['WIND_DIRECTION'].count()} 个有效值")
                        logger.debug(f"处理风速数据: 提取了 {cleaned['WIND_SPEED'].count()} 个有效值")
                    else:
                        logger.warning(f"WND列格式不正确，无法正确分割: {wnd_parts.iloc[0] if len(wnd_parts) > 0 else 'Empty'}")
                except Exception as e:
                    logger.error(f"处理风速数据时出错: {e}")
            else:
                logger.warning("数据中缺少WND列，无法处理风向风速数据")
            
            # 5. 计算相对湿度 (根据气温和露点温度)
            if 'TEMPERATURE' in cleaned.columns and 'DEWPOINT' in cleaned.columns:
                try:
                    # 使用公式计算相对湿度
                    # RH = 100 * exp((17.625 * Td)/(243.04 + Td)) / exp((17.625 * T)/(243.04 + T))
                    # 其中T是气温，Td是露点温度，均为摄氏度
                    
                    def calculate_rh(row):
                        t = row['TEMPERATURE']
                        td = row['DEWPOINT']
                        
                        if pd.isna(t) or pd.isna(td):
                            return np.nan
                            
                        # 如果露点温度大于气温，有误差，将露点温度设为气温
                        if td > t:
                            td = t
                            
                        # 计算相对湿度
                        es_t = 6.112 * math.exp((17.67 * t) / (t + 243.5))
                        es_td = 6.112 * math.exp((17.67 * td) / (td + 243.5))
                        rh = 100.0 * es_td / es_t
                        
                        # 限制在0-100范围内
                        return min(max(rh, 0), 100)
                    
                    # 应用函数计算相对湿度
                    cleaned['HUMIDITY'] = cleaned.apply(calculate_rh, axis=1)
                    # 打印调试信息
                    logger.debug(f"计算相对湿度: 计算出 {cleaned['HUMIDITY'].count()} 个有效值")
                except Exception as e:
                    logger.error(f"计算相对湿度时出错: {e}")
                    # 创建空的相对湿度列
                    cleaned['HUMIDITY'] = np.nan
            else:
                if 'TEMPERATURE' not in cleaned.columns:
                    logger.warning("数据中缺少TEMPERATURE列，无法计算相对湿度")
                if 'DEWPOINT' not in cleaned.columns:
                    logger.warning("数据中缺少DEWPOINT列，无法计算相对湿度")
            
            # 检查是否成功提取了任何数据
            data_columns = ['TEMPERATURE', 'DEWPOINT', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION']
            has_data = False
            for col in data_columns:
                if col in cleaned.columns and cleaned[col].count() > 0:
                    has_data = True
                    break
            
            if not has_data:
                logger.warning("数据清洗后未提取出任何有效的气象数据")
                
            # 保留需要的列
            essential_columns = ['STATION', 'DATE', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION']
            data_columns = ['TEMPERATURE', 'DEWPOINT', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION']
            
            # 构建最终的列列表（仅包含存在的列）
            final_columns = [col for col in essential_columns + data_columns if col in cleaned.columns]
            
            # 打印返回数据的基本信息
            valid_data = cleaned[final_columns].copy()
            logger.debug(f"数据清洗完成，返回 {len(valid_data)} 行数据，包含以下列: {', '.join(valid_data.columns)}")
            for col in data_columns:
                if col in valid_data.columns:
                    logger.debug(f"  - {col}: {valid_data[col].count()} 个非空值")
            
            return valid_data
            
        except Exception as e:
            logger.error(f"数据清洗过程中出错: {e}")
            return df
    
    def save_to_excel(self, df, sheet_name=None):
        """
        将数据保存到Excel文件，每个变量作为一个单独的sheet
        参数:
            df: 要保存的DataFrame
            sheet_name: 可选的sheet名称
        """
        if df.empty:
            logger.warning("没有数据需要保存")
            return

        try:
            # 检查必要的列是否存在
            if 'STATION' not in df.columns or 'NAME' not in df.columns or 'DATE' not in df.columns:
                logger.warning("输入数据中缺少必要的列(STATION, NAME, DATE)")
                if 'DATE' in df.columns:
                    df.to_excel(self.output_file, sheet_name=sheet_name or "气象数据", index=False)
                    logger.info(f"数据已成功保存到 {self.output_file}")
                return

            logger.info(f"开始处理数据并按变量保存为Excel文件...")
            
            # 确保DATE列是datetime类型
            df['DATE'] = pd.to_datetime(df['DATE'])
            
            # 打印数据概况
            logger.info(f"数据概况：共 {len(df)} 行数据，{df['STATION'].nunique()} 个站点")
            logger.debug(f"日期范围：{df['DATE'].min()} 到 {df['DATE'].max()}")
            
            # 创建站点映射（ID -> 名称）
            stations_map = {}
            for _, row in df.drop_duplicates(['STATION', 'NAME']).iterrows():
                station_id = str(row['STATION'])
                # 处理站点名称，去掉逗号后的部分
                station_name = row['NAME'].split(',')[0] if isinstance(row['NAME'], str) and ',' in row['NAME'] else row['NAME']
                stations_map[station_id] = station_name
            
            logger.debug(f"找到 {len(stations_map)} 个站点: {', '.join(stations_map.values())}")
            
            # 生成完整的时间序列（按小时）
            start_time_dt = pd.to_datetime(START_DATE).replace(hour=0, minute=0, second=0)
            end_time_dt = pd.to_datetime(END_DATE).replace(hour=23, minute=0, second=0)
            full_date_range = pd.date_range(start=start_time_dt, end=end_time_dt, freq='h')
            logger.info(f"生成完整时间序列: {start_time_dt} 到 {end_time_dt}，共 {len(full_date_range)} 个时间点")
            
            # 定义要处理的列（变量）
            data_columns = ['TEMPERATURE', 'DEWPOINT', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION']
            # 过滤出实际存在的数据列
            available_data_columns = [col for col in data_columns if col in df.columns]
            
            logger.info(f"要处理的变量列: {', '.join(available_data_columns)}")
            
            # 检查每个变量是否有数据
            logger.debug("变量数据统计:")
            for col in available_data_columns:
                non_null_count = df[col].count()
                logger.debug(f"  - {col}: {non_null_count} 个非空值，占比 {non_null_count*100/len(df):.2f}%")
            
            # 创建ExcelWriter对象
            with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                # 保存站点信息
                if self.filtered_stations is not None and not self.filtered_stations.empty:
                    # 只选择有数据的站点
                    station_ids_in_data = list(stations_map.keys())
                    
                    # 从站点ID中提取标准格式的ID
                    processed_station_ids = []
                    for sid in station_ids_in_data:
                        # 如果站点ID以5开头且长度大于5，取前5位
                        if sid.startswith('5') and len(sid) > 5:
                            processed_station_ids.append(sid[:5])
                        else:
                            processed_station_ids.append(sid)
                    
                    logger.debug(f"处理后的站点ID列表: {', '.join(processed_station_ids)}")
                    
                    # 筛选有数据的站点
                    stations_with_data_df = self.filtered_stations[
                        self.filtered_stations['区站号'].isin(processed_station_ids)
                    ].copy()
                    
                    if stations_with_data_df.empty:
                        logger.warning(f"在站点信息表中未找到匹配的站点（检查依据: {', '.join(processed_station_ids)}）")
                        stations_df = pd.DataFrame({
                            'Station ID': station_ids_in_data,
                            'Station Name': [stations_map[sid] for sid in station_ids_in_data]
                        })
                    else:
                        # 选择需要的列并重命名
                        columns_to_save = [
                            '区站号', '站名', 'Station_Name_EN', '纬度(十进制度)', '经度(十进制度)', 
                            '海拔高度(m)', 'Elevation_CSV', '国别', '省份'
                        ]
                        
                        # 只保留存在的列
                        available_columns = [col for col in columns_to_save if col in stations_with_data_df.columns]
                        
                        # 创建保存的DataFrame
                        stations_df = stations_with_data_df[available_columns].copy()
                        
                        # 重命名列以便于理解
                        rename_dict = {
                            '区站号': 'Station ID',
                            '站名': 'Chinese Name',
                            'Station_Name_EN': 'English Name',
                            '纬度(十进制度)': 'Latitude',
                            '经度(十进制度)': 'Longitude',
                            '海拔高度(m)': 'Elevation (m)',
                            'Elevation_CSV': 'Elevation from CSV (m)',
                            '国别': 'Country',
                            '省份': 'Province'
                        }
                        
                        # 只重命名存在的列
                        rename_columns = {k: v for k, v in rename_dict.items() if k in stations_df.columns}
                        if rename_columns:
                            stations_df = stations_df.rename(columns=rename_columns)
                else:
                    # 如果没有筛选后的站点信息，创建一个简单的站点信息表
                    stations_df = pd.DataFrame({
                        'Station ID': list(stations_map.keys()),
                        'Station Name': list(stations_map.values())
                    })
                
                # 保存站点信息表
                try:
                    stations_df.to_excel(writer, sheet_name='stations', index=False)
                    logger.info(f"已将站点信息保存到sheet 'stations'，包含 {len(stations_df)} 行")
                except Exception as e:
                    logger.error(f"保存站点信息到stations sheet时出错: {e}")
                
                # 针对每个变量创建一个sheet
                logger.info("开始创建变量数据表...")
                saved_var_count = 0
                
                # 针对每个变量创建一个sheet，列为不同站点
                for variable in available_data_columns:
                    logger.debug(f"处理变量: {variable}")
                    
                    # 创建包含完整时间序列的DataFrame
                    var_df = pd.DataFrame(index=full_date_range)
                    
                    # 记录有数据的站点数
                    stations_with_data_count = 0
                    
                    # 检查这个变量是否有数据
                    if df[variable].count() == 0:
                        logger.debug(f"变量 {variable} 没有任何非空数据")
                        # 依然保存空表
                        var_df.to_excel(writer, sheet_name=variable)
                        logger.debug(f"已将空的 {variable} 数据表保存到sheet '{variable}'")
                        continue
                    
                    # 为每个站点添加数据列
                    for station_id, station_name in stations_map.items():
                        # 获取该站点的数据
                        station_df = df[df['STATION'] == station_id].copy()
                        
                        # 检查该站点是否有这个变量的数据
                        if len(station_df) > 0 and variable in station_df.columns and station_df[variable].count() > 0:
                            # 按时间排序并去重，保留第一次出现的记录
                            station_df = station_df.sort_values('DATE').drop_duplicates('DATE', keep='first')
                            
                            # 设置日期为索引并选择变量列
                            station_var_df = station_df.set_index('DATE')[variable]
                            
                            # 将站点数据添加到变量DataFrame
                            var_df[station_name] = station_var_df
                            
                            # 计算非空值数量
                            non_null_count = station_var_df.count()
                            
                            if non_null_count > 0:
                                logger.debug(f"  - 站点 {station_name} 在 {variable} 中有 {non_null_count} 个非空值")
                                stations_with_data_count += 1
                    
                    # 如果没有任何站点有数据，提供警告
                    if stations_with_data_count == 0:
                        logger.debug(f"变量 {variable} 在所有站点中都没有有效数据")
                    else:
                        logger.debug(f"变量 {variable} 共有 {stations_with_data_count} 个站点有数据")
                    
                    # 保存变量数据到对应sheet
                    try:
                        var_df.to_excel(writer, sheet_name=variable)
                        saved_var_count += 1
                        logger.debug(f"已将变量 {variable} 的数据保存到sheet '{variable}'")
                    except Exception as e:
                        logger.error(f"保存变量 {variable} 到sheet时出错: {e}")
                
                logger.info(f"成功保存了 {saved_var_count} 个变量数据表")
            
            logger.info(f"数据已成功保存到 {self.output_file}")

        except Exception as e:
            logger.error(f"保存数据到Excel时出错: {e}")
            raise
            
    def extract_data(self, start_date=None, end_date=None, 
                    station_names=None, station_ids=None,
                    min_lat=None, max_lat=None, min_lon=None, max_lon=None,
                    clean_data=True, filter_mode='coords'):
        """
        从NCDC数据中提取指定条件的气象数据
        
        参数:
            start_date: 起始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            station_names: 站点名称列表
            station_ids: 站点ID列表
            min_lat: 最小纬度
            max_lat: 最大纬度
            min_lon: 最小经度
            max_lon: 最大经度
            clean_data: 是否清洗数据
            filter_mode: 当同时设置经纬度范围和站点列表时的过滤模式（'coords'或'stations'）
            
        返回:
            pandas.DataFrame: 提取的数据
        """
        # 加载站点信息
        self._load_station_info()
        
        # 转换日期为datetime对象
        if start_date:
            start_date = pd.to_datetime(start_date).replace(hour=0, minute=0, second=0)
        if end_date:
            end_date = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
            
        logger.info(f"检索时间范围: {start_date} 至 {end_date}")
        
        # 根据经纬度范围筛选站点
        if min_lat is not None and max_lat is not None and min_lon is not None and max_lon is not None:
            logger.info(f"使用经纬度范围筛选站点: {min_lat}-{max_lat}°N, {min_lon}-{max_lon}°E")
            # 确保站点数据为深拷贝，避免SettingWithCopyWarning
            self.filtered_stations = filter_stations_by_coords(
                self.station_info, min_lat, max_lat, min_lon, max_lon
            ).copy()  # 添加.copy()确保是深拷贝
            
            if self.filtered_stations.empty:
                logger.warning("在指定经纬度范围内未找到任何站点")
                return pd.DataFrame()
                
            # 获取筛选后的站点ID列表
            filtered_station_ids = self.filtered_stations['区站号'].tolist()  # 已在_load_station_info中转换为字符串
            logger.info(f"在指定范围内找到 {len(filtered_station_ids)} 个站点")
            
            # 查找站点对应的CSV文件，指定YEAR_LIMIT为2024
            logger.info("查找站点对应的CSV文件...")
            year_limit = "2024"  # 指定查找2024年的数据
            csv_files, stations_with_data = find_csv_files_for_stations(
                filtered_station_ids, str(self.data_dir), year_limit
            )
            
            if not stations_with_data:
                logger.warning("未找到任何站点的CSV数据文件")
                return pd.DataFrame()
                
            logger.info(f"共有 {len(stations_with_data)}/{len(filtered_station_ids)} 个站点找到对应的CSV文件")
            
            # 从CSV文件获取英文名称和高度信息
            station_english_names = {}
            station_elevations = {}
            
            for station_id in stations_with_data:
                if station_id in csv_files and csv_files[station_id]:
                    # 只读取第一个CSV文件的站点信息
                    file_path = csv_files[station_id][0]
                    logger.debug(f"从文件获取站点信息: {file_path}")
                    
                    try:
                        # 只读取第一行获取站点名称和高度
                        df_header = pd.read_csv(file_path, nrows=1)
                        
                        if 'NAME' in df_header.columns:
                            name = df_header['NAME'].iloc[0]
                            # 处理站点名称，将逗号分隔的部分的第一部分作为站点英文名
                            english_name = name.split(',')[0] if ',' in name else name
                            station_english_names[station_id] = english_name
                            logger.debug(f"获取到站点 {station_id} 的英文名称: {english_name}")
                        
                        if 'ELEVATION' in df_header.columns:
                            elevation = df_header['ELEVATION'].iloc[0]
                            station_elevations[station_id] = elevation
                            logger.debug(f"获取到站点 {station_id} 的高度: {elevation}")
                    except Exception as e:
                        logger.error(f"从文件 {file_path} 获取站点信息时出错: {e}")
            
            # 将英文名称和高度信息添加到筛选后的站点中
            if station_english_names or station_elevations:
                # 创建英文名称列
                if station_english_names:
                    # 使用.loc避免SettingWithCopyWarning
                    self.filtered_stations['Station_Name_EN'] = [
                        station_english_names.get(str(sid), "Unknown") 
                        for sid in self.filtered_stations['区站号']
                    ]
                
                # 创建高度列
                if station_elevations:
                    # 使用.loc避免SettingWithCopyWarning
                    self.filtered_stations['Elevation_CSV'] = [
                        station_elevations.get(str(sid), None) 
                        for sid in self.filtered_stations['区站号']
                    ]
            
            # 读取并处理每个站点的数据
            logger.info("开始读取和处理站点数据...")
            all_data = []
            
            # 记录处理进度
            total_stations = len(stations_with_data)
            processed_count = 0
            
            for station_id in stations_with_data:
                station_files = csv_files[station_id]
                for file_path in station_files:
                    # 使用DEBUG级别不显示在控制台，但保存在日志文件中
                    logger.debug(f"处理文件: {file_path}")
                    df = self._parse_csv_data(file_path)
                    
                    if df.empty:
                        logger.debug(f"文件 {file_path} 未读取到数据")
                        continue
                    
                    # 检查这个CSV文件中的STATION ID是什么，以便确保使用正确的ID
                    actual_station_id = None
                    if 'STATION' in df.columns:
                        actual_station_id = str(df['STATION'].iloc[0])
                        logger.debug(f"文件中的站点ID: {actual_station_id}，与查找ID {station_id} 比较")
                    
                    # 按日期范围过滤
                    if 'DATE' in df.columns:
                        if start_date and end_date:
                            logger.debug(f"按日期范围过滤: {start_date} 至 {end_date}")
                            # 确保DATE是datetime类型
                            df['DATE'] = pd.to_datetime(df['DATE'])
                            # 过滤指定日期范围内的数据
                            filtered_df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
                            logger.debug(f"过滤前记录数: {len(df)}, 过滤后记录数: {len(filtered_df)}")
                            df = filtered_df
                        
                    if not df.empty:
                        # 确保使用正确的站点ID
                        if actual_station_id is not None and 'STATION' in df.columns:
                            df['STATION'] = actual_station_id
                        else:
                            logger.debug(f"文件 {file_path} 中找不到STATION列，或未能确定正确的站点ID")
                        
                        all_data.append(df)
                
                # 更新进度
                processed_count += 1
                if processed_count % 10 == 0 or processed_count == total_stations:
                    logger.info(f"已处理 {processed_count}/{total_stations} 个站点")
            
            # 合并所有数据
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                logger.info(f"合并后共有 {len(result)} 条记录")
                
                # 数据清洗
                if clean_data:
                    logger.info("开始清洗数据...")
                    # 获取清洗前各变量的数据统计
                    data_columns = ['TMP', 'DEW', 'SLP', 'WND']
                    for col in data_columns:
                        if col in result.columns:
                            col_count = result[col].count()
                            logger.debug(f"清洗前变量 {col} 有 {col_count} 个非空值")
                    
                    # 清洗数据
                    cleaned_result = self._clean_data(result)
                    
                    # 获取清洗后各变量的数据统计
                    data_columns = ['TEMPERATURE', 'DEWPOINT', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION']
                    logger.info("清洗后变量统计:")
                    for col in data_columns:
                        if col in cleaned_result.columns:
                            col_count = cleaned_result[col].count()
                            logger.info(f"  - {col}: {col_count} 个非空值")
                    
                    result = cleaned_result
                    
                logger.info(f"成功提取了 {len(result)} 条记录")
                return result
            else:
                logger.warning("没有找到符合条件的数据")
                return pd.DataFrame()
        else:
            logger.error("必须提供经纬度范围参数才能提取数据")
            return pd.DataFrame()

def main():
    """主函数"""
    # 在函数最开始输出关键信息
    logger.info("========== NCDC气象数据提取开始 ==========")
    logger.info(f"数据目录: {DATA_DIR}")
    logger.info(f"输出文件: {OUTPUT_FILE}")
    logger.info(f"时间范围: {START_DATE} 至 {END_DATE}")
    logger.info(f"经纬度范围: {MIN_LAT}-{MAX_LAT}°N, {MIN_LON}-{MAX_LON}°E")
    logger.info("========================================")
    
    start_time = time.time()
    
    try:
        # 创建提取器
        extractor = NCDCExtractor(
            data_dir=DATA_DIR,
            output_file=OUTPUT_FILE
        )
        
        # 提取数据
        logger.info("开始提取数据...")
        df = extractor.extract_data(
            start_date=START_DATE,
            end_date=END_DATE,
            station_names=STATION_NAMES,
            station_ids=STATION_IDS,
            min_lat=MIN_LAT,
            max_lat=MAX_LAT,
            min_lon=MIN_LON,
            max_lon=MAX_LON,
            clean_data=CLEAN_DATA,
            filter_mode=FILTER_MODE
        )
        
        # 保存到Excel
        if not df.empty:
            logger.info("正在保存数据到Excel...")
            extractor.save_to_excel(df)
            logger.info(f"处理完成，数据已保存到 {OUTPUT_FILE}")
        else:
            # 即使没有找到数据，也创建包含均匀时间序列的空表格
            logger.warning("没有找到符合条件的数据，将创建包含时间序列的空表格")
            
            # 创建ExcelWriter对象
            with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
                # 保存站点信息
                if extractor.filtered_stations is not None and not extractor.filtered_stations.empty:
                    logger.info(f"保存筛选后的 {len(extractor.filtered_stations)} 个站点信息到Excel")
                    
                    # 选择需要的列
                    columns_to_save = [
                        '区站号', '站名', '纬度(十进制度)', '经度(十进制度)', 
                        '海拔高度(m)', '国别', '省份'
                    ]
                    
                    # 只保留存在的列
                    available_columns = [col for col in columns_to_save if col in extractor.filtered_stations.columns]
                    
                    # 创建保存的DataFrame
                    stations_df = extractor.filtered_stations[available_columns].copy()
                    
                    # 重命名列以便于理解
                    rename_dict = {
                        '区站号': 'Station ID',
                        '站名': 'Chinese Name',
                        '纬度(十进制度)': 'Latitude',
                        '经度(十进制度)': 'Longitude',
                        '海拔高度(m)': 'Elevation (m)',
                        '国别': 'Country',
                        '省份': 'Province'
                    }
                    
                    # 只重命名存在的列
                    rename_columns = {k: v for k, v in rename_dict.items() if k in stations_df.columns}
                    if rename_columns:
                        stations_df = stations_df.rename(columns=rename_columns)
                    
                    # 保存站点信息
                    stations_df.to_excel(writer, sheet_name='stations', index=False)
                    logger.info(f"已将站点信息保存到sheet 'stations'，包含 {len(stations_df)} 行")
                else:
                    # 创建空的站点信息表
                    pd.DataFrame(columns=['Station ID', 'Chinese Name', 'Latitude', 'Longitude']).to_excel(
                        writer, sheet_name='stations', index=False
                    )
                    logger.info("已创建空的站点信息表")
                
                # 生成完整的时间序列（按小时）
                start_time_dt = pd.to_datetime(START_DATE).replace(hour=0, minute=0, second=0)
                end_time_dt = pd.to_datetime(END_DATE).replace(hour=23, minute=0, second=0)
                full_date_range = pd.date_range(start=start_time_dt, end=end_time_dt, freq='h')
                logger.info(f"生成完整时间序列: {start_time_dt} 到 {end_time_dt}，共 {len(full_date_range)} 个时间点")
                
                # 创建一个空的DataFrame，包含完整的时间序列
                time_df = pd.DataFrame({'DATE': full_date_range})
                time_df.set_index('DATE', inplace=True)
                
                # 定义气象要素变量列表
                variables = ['TEMPERATURE', 'DEWPOINT', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION']
                
                # 为每个变量创建一个空的sheet
                for variable in variables:
                    # 保存变量数据到对应sheet
                    try:
                        time_df.to_excel(writer, sheet_name=variable)
                        logger.info(f"已将空的 {variable} 数据表保存到sheet '{variable}'")
                    except Exception as e:
                        logger.error(f"保存变量 {variable} 到sheet时出错: {e}")
            
            logger.info(f"已创建包含时间序列的空表格，保存到 {OUTPUT_FILE}")
        
        # 计算总耗时
        total_time = time.time() - start_time
        logger.info(f"总耗时: {total_time:.2f}秒")
        logger.info("========== NCDC气象数据提取完成 ==========")
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 