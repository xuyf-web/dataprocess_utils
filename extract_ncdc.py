#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-

# ============================================================
# 参数配置区域（根据需要修改以下参数）
# ============================================================

# 输入/输出参数
DATA_DIR = '/data8/xuyf/data/obs/meteo_ncdc/2024'  # NCDC数据所在目录
OUTPUT_FILE = './output/shouxian202404.xlsx'    # 输出Excel文件路径

# 时间范围参数
START_DATE = '2024-04-01'        # 起始日期，格式：'YYYY-MM-DD'
END_DATE = '2024-04-15'          # 结束日期，格式：'YYYY-MM-DD'

# 空间范围选择参数
# 当同时设置了经纬度范围和站点列表时，此参数决定使用哪种方式过滤数据
# 可选值：'coords'（使用经纬度范围）或 'stations'（使用站点列表）
FILTER_MODE = 'coords'

# 经纬度范围参数
MIN_LAT = 30.0                   # 最小纬度
MAX_LAT = 35.0                   # 最大纬度
MIN_LON = 114.0                  # 最小经度
MAX_LON = 122.0                  # 最大经度

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
LOG_FILE = "ncdc_extract.log"

# 删除已有的日志文件
if os.path.exists(LOG_FILE):
    try:
        os.remove(LOG_FILE)
        print(f"已删除旧的日志文件: {LOG_FILE}")
    except Exception as e:
        print(f"删除旧的日志文件时出错: {e}")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
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
                except Exception as e:
                    logger.error(f"处理气温数据时出错: {e}")
            
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
                except Exception as e:
                    logger.error(f"处理露点温度数据时出错: {e}")
            
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
                except Exception as e:
                    logger.error(f"处理气压数据时出错: {e}")
            
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
                except Exception as e:
                    logger.error(f"处理风速数据时出错: {e}")
            
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
                except Exception as e:
                    logger.error(f"计算相对湿度时出错: {e}")
                    # 创建空的相对湿度列
                    cleaned['HUMIDITY'] = np.nan
            
            # 保留需要的列
            essential_columns = ['STATION', 'DATE', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION']
            data_columns = ['TEMPERATURE', 'DEWPOINT', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION']
            
            # 构建最终的列列表（仅包含存在的列）
            final_columns = [col for col in essential_columns + data_columns if col in cleaned.columns]
            
            return cleaned[final_columns]
            
        except Exception as e:
            logger.error(f"数据清洗过程中出错: {e}")
            return df
    
    def save_to_excel(self, df, sheet_name=None):
        """
        将数据保存到Excel文件，每个站点作为一个单独的sheet
        参数:
            df: 要保存的DataFrame
            sheet_name: 可选的sheet名称
        """
        if df.empty:
            logger.warning("没有数据需要保存")
            return

        try:
            # 按站点分组
            if 'STATION' in df.columns and 'NAME' in df.columns:
                # 为每个站点创建一个单独的DataFrame
                grouped = df.groupby(['STATION', 'NAME'])
                logger.info(f"数据包含 {len(grouped)} 个站点组")

                # 获取有数据的站点ID列表
                stations_with_data = list(set([str(station_id) for (station_id, _) in grouped.groups.keys()]))
                logger.info(f"生成Excel文件，包含 {len(stations_with_data)} 个有数据的站点: {', '.join(stations_with_data)}")

                # 创建ExcelWriter对象
                with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                    # 创建一个DataFrame并将筛选后的站点信息写入stations sheet
                    if self.filtered_stations is not None and not self.filtered_stations.empty:
                        logger.info(f"筛选后的站点表包含 {len(self.filtered_stations)} 个站点")
                        
                        # 确保区站号列是字符串类型
                        self.filtered_stations['区站号'] = self.filtered_stations['区站号'].astype(str)
                        
                        # 打印所有站点ID用于调试
                        all_station_ids = self.filtered_stations['区站号'].tolist()
                        logger.info(f"筛选后的站点ID: {', '.join(all_station_ids[:20])}...")
                        
                        # 提取CSV文件名中的站点ID（去掉文件后缀0）
                        station_ids_in_data = [s[:5] if s.startswith('5') else s for s in stations_with_data]
                        logger.info(f"有数据的站点ID (处理后): {', '.join(station_ids_in_data[:20])}...")
                        
                        # 只选择有数据的站点
                        stations_with_data_df = self.filtered_stations[
                            self.filtered_stations['区站号'].isin(station_ids_in_data)
                        ].copy()
                        
                        logger.info(f"找到 {len(stations_with_data_df)} 个有数据的站点")
                        
                        if stations_with_data_df.empty:
                            logger.warning("没有找到有数据的站点信息")
                            stations_df = pd.DataFrame(columns=['Station ID', 'Chinese Name', 'English Name', 'Latitude', 'Longitude', 'Elevation (m)', 'Elevation from CSV (m)', 'Country', 'Province'])
                        else:
                            # 选择需要的列
                            columns_to_save = [
                                '区站号', '站名', 'Station_Name_EN', '纬度(十进制度)', '经度(十进制度)', 
                                '海拔高度(m)', 'Elevation_CSV', '国别', '省份'
                            ]
                            
                            # 只保留存在的列
                            available_columns = [col for col in columns_to_save if col in stations_with_data_df.columns]
                            logger.info(f"使用以下列: {', '.join(available_columns)}")
                            
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
                                
                            logger.info(f"stations表包含 {len(stations_df)} 个有数据的站点，列: {', '.join(stations_df.columns.tolist())}")
                    else:
                        logger.warning("筛选后的站点表为空")
                        # 如果没有筛选后的站点信息，创建空的DataFrame
                        stations_df = pd.DataFrame(columns=['Station ID', 'Chinese Name', 'English Name', 'Latitude', 'Longitude', 'Elevation (m)', 'Elevation from CSV (m)', 'Country', 'Province'])
                    
                    # 首先将站点信息写入名为'stations'的sheet（放在最前面）
                    try:
                        stations_df.to_excel(writer, sheet_name='stations', index=False)
                        logger.info(f"已将站点信息保存到sheet 'stations'，包含 {len(stations_df)} 行")
                    except Exception as e:
                        logger.error(f"保存站点信息到stations sheet时出错: {e}")

                    # 然后将每个站点的数据保存到单独的sheet
                    for (station_id, station_name), station_df in grouped:
                        # 处理站点名称，去掉逗号后的部分
                        sheet_name = station_name.split(',')[0]  # 只保留逗号前的部分

                        # 设置日期为索引
                        if 'DATE' in station_df.columns:
                            station_df = station_df.set_index('DATE')

                        # 移除不需要的列
                        drop_cols = ['STATION', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION']
                        station_df = station_df.drop(columns=[col for col in drop_cols if col in station_df.columns])

                        # 保存到sheet
                        try:
                            station_df.to_excel(writer, sheet_name=sheet_name)
                            logger.info(f"已将站点 {station_id} ({station_name}) 的数据保存到sheet '{sheet_name}'")
                        except Exception as e:
                            logger.error(f"保存站点 {station_id} 数据到sheet时出错: {e}")

                logger.info(f"数据已成功保存到 {self.output_file}")
            else:
                # 如果没有站点和名称列，就保存为一个单独的sheet
                logger.warning("输入数据中缺少STATION或NAME列")
                df.to_excel(self.output_file, sheet_name=sheet_name or "气象数据", index=False)
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
            start_date = pd.to_datetime(start_date)
        if end_date:
            end_date = pd.to_datetime(end_date)
            
        logger.info(f"检索时间范围: {start_date} 至 {end_date}")
        
        # 根据经纬度范围筛选站点
        if min_lat is not None and max_lat is not None and min_lon is not None and max_lon is not None:
            logger.info(f"使用经纬度范围筛选站点: {min_lat}-{max_lat}°N, {min_lon}-{max_lon}°E")
            self.filtered_stations = filter_stations_by_coords(
                self.station_info, min_lat, max_lat, min_lon, max_lon
            )
            
            if self.filtered_stations.empty:
                logger.warning("在指定经纬度范围内未找到任何站点")
                return pd.DataFrame()
                
            # 获取筛选后的站点ID列表
            filtered_station_ids = self.filtered_stations['区站号'].astype(str).tolist()
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
                    logger.info(f"从文件获取站点信息: {file_path}")
                    
                    try:
                        # 只读取第一行获取站点名称和高度
                        df_header = pd.read_csv(file_path, nrows=1)
                        
                        if 'NAME' in df_header.columns:
                            name = df_header['NAME'].iloc[0]
                            # 处理站点名称，将逗号分隔的部分的第一部分作为站点英文名
                            english_name = name.split(',')[0] if ',' in name else name
                            station_english_names[station_id] = english_name
                            logger.info(f"获取到站点 {station_id} 的英文名称: {english_name}")
                        
                        if 'ELEVATION' in df_header.columns:
                            elevation = df_header['ELEVATION'].iloc[0]
                            station_elevations[station_id] = elevation
                            logger.info(f"获取到站点 {station_id} 的高度: {elevation}")
                    except Exception as e:
                        logger.error(f"从文件 {file_path} 获取站点信息时出错: {e}")
            
            # 将英文名称和高度信息添加到筛选后的站点中
            if station_english_names or station_elevations:
                # 创建英文名称列
                if station_english_names:
                    self.filtered_stations['Station_Name_EN'] = self.filtered_stations['区站号'].astype(str).map(
                        lambda x: station_english_names.get(x, "Unknown")
                    )
                
                # 创建高度列
                if station_elevations:
                    self.filtered_stations['Elevation_CSV'] = self.filtered_stations['区站号'].astype(str).map(
                        lambda x: station_elevations.get(x, None)
                    )
            
            # 读取并处理每个站点的数据
            all_data = []
            for station_id in stations_with_data:
                station_files = csv_files[station_id]
                for file_path in station_files:
                    logger.info(f"处理文件: {file_path}")
                    df = self._parse_csv_data(file_path)
                    
                    if df.empty:
                        continue
                    
                    # 按日期范围过滤
                    if 'DATE' in df.columns:
                        if start_date and end_date:
                            logger.info(f"按日期范围过滤: {start_date} 至 {end_date}")
                            # 确保DATE是datetime类型
                            df['DATE'] = pd.to_datetime(df['DATE'])
                            # 过滤指定日期范围内的数据
                            filtered_df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
                            logger.info(f"过滤前记录数: {len(df)}, 过滤后记录数: {len(filtered_df)}")
                            df = filtered_df
                        
                    if not df.empty:
                        all_data.append(df)
            
            # 合并所有数据
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                
                # 数据清洗
                if clean_data:
                    result = self._clean_data(result)
                    
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
    logger.info("开始提取NCDC气象数据")
    logger.info(f"数据目录: {DATA_DIR}")
    logger.info(f"输出文件: {OUTPUT_FILE}")
    logger.info(f"时间范围: {START_DATE} 至 {END_DATE}")
    logger.info(f"经纬度范围: {MIN_LAT}-{MAX_LAT}°N, {MIN_LON}-{MAX_LON}°E")
    
    start_time = time.time()
    
    try:
        # 创建提取器
        extractor = NCDCExtractor(
            data_dir=DATA_DIR,
            output_file=OUTPUT_FILE
        )
        
        # 提取数据
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
            extractor.save_to_excel(df)
            logger.info(f"处理完成，数据已保存到 {OUTPUT_FILE}")
        else:
            # 即使没有找到数据，也尝试保存筛选后的站点信息
            logger.warning("没有找到符合条件的数据")
            
            # 直接将筛选后的所有站点信息保存到Excel
            if extractor.filtered_stations is not None and not extractor.filtered_stations.empty:
                logger.info(f"仍然保存筛选后的 {len(extractor.filtered_stations)} 个站点信息到Excel")
                
                # 选择需要的列
                columns_to_save = [
                    '区站号', '站名', '纬度(十进制度)', '经度(十进制度)', 
                    '海拔高度(m)', '国别', '省份'
                ]
                
                # 只保留存在的列
                available_columns = [col for col in columns_to_save if col in extractor.filtered_stations.columns]
                logger.info(f"使用以下列: {', '.join(available_columns)}")
                
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
                
                # 直接保存为Excel
                stations_df.to_excel(OUTPUT_FILE, sheet_name='stations', index=False)
                logger.info(f"已将站点信息保存到 {OUTPUT_FILE}, sheet: stations，包含 {len(stations_df)} 行")
            else:
                logger.warning(f"未生成Excel文件，因为没有找到符合条件的站点")
        
        # 计算总耗时
        total_time = time.time() - start_time
        logger.info(f"总耗时: {total_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 