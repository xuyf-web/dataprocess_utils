#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-

# ============================================================
# 参数配置区域（根据需要修改以下参数）
# ============================================================

# 输入/输出参数
DATA_DIR = '/data8/xuyf/data/obs/chem_cn_wxl/sites_20240101-20241207'  # 数据所在目录
OUTPUT_FILE = '/data8/xuyf/data/obs/chem_cn_wxl/output/xishan2024.xlsx'    # 输出Excel文件路径
SITE_LOCATION_FILE = '/data8/xuyf/data/obs/chem_cn_wxl/sitelocations_from2022.02.13.xlsx'  # 站点位置文件

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

# 站点列表参数 (站点ID格式如: '1001A', '1002A')
STATION_IDS = None               # 站点ID列表，例如：['1001A', '1002A']
STATION_NAMES = None             # 站点名称列表，例如：['北京', '上海']

# ============================================================
# 以下是脚本代码，一般不需要修改
# ============================================================

import os
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import glob
import logging
import time

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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建输出目录
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

class ChemExtractor:
    """
    化学观测数据提取器
    用于按照时间范围和空间范围提取化学观测数据并保存为Excel文件
    """
    
    def __init__(self, data_dir, output_file, site_location_file):
        """
        初始化提取器
        
        参数:
            data_dir: 化学观测数据所在目录
            output_file: 输出Excel文件路径
            site_location_file: 站点位置信息文件
        """
        self.data_dir = Path(data_dir)
        self.output_file = output_file
        self.site_location_file = site_location_file
        self.station_info = None
        self.filtered_stations = None
    
    def _load_station_info(self):
        """加载站点信息"""
        try:
            # 尝试从站点信息文件加载站点信息
            station_file = Path(self.site_location_file)
            if station_file.exists():
                self.station_info = pd.read_excel(station_file)
                logger.info(f"成功加载了 {len(self.station_info)} 个站点信息")
            else:
                logger.warning(f"警告: 找不到站点信息文件 {station_file}")
                self.station_info = pd.DataFrame()
        except Exception as e:
            logger.error(f"加载站点信息时出错: {e}")
            self.station_info = pd.DataFrame()
    
    def _filter_stations_by_coords(self, min_lat, max_lat, min_lon, max_lon):
        """
        根据经纬度范围筛选站点
        
        参数:
            min_lat: 最小纬度
            max_lat: 最大纬度
            min_lon: 最小经度
            max_lon: 最大经度
        
        返回:
            筛选后的站点ID列表
        """
        if self.station_info is None:
            self._load_station_info()
        
        if self.station_info.empty:
            logger.warning("站点信息为空，无法根据经纬度筛选站点")
            return []
        
        # 检查站点信息表中是否有经纬度列
        lat_col = None
        lon_col = None
        
        # 常见的经纬度列名
        lat_candidates = ['lat', 'latitude', 'Lat', 'Latitude', '纬度', 'LAT']
        lon_candidates = ['lon', 'longitude', 'Lon', 'Longitude', '经度', 'LON']
        
        for col in self.station_info.columns:
            if col in lat_candidates:
                lat_col = col
            if col in lon_candidates:
                lon_col = col
        
        if lat_col is None or lon_col is None:
            logger.warning("站点信息表中没有找到经纬度列，无法根据经纬度筛选站点")
            return []
        
        # 确保经纬度列是数值类型
        try:
            self.station_info[lat_col] = pd.to_numeric(self.station_info[lat_col], errors='coerce')
            self.station_info[lon_col] = pd.to_numeric(self.station_info[lon_col], errors='coerce')
        except Exception as e:
            logger.error(f"转换经纬度列为数值类型时出错: {e}")
            return []
        
        # 根据经纬度范围筛选站点
        filtered = self.station_info[
            (self.station_info[lat_col] >= min_lat) & 
            (self.station_info[lat_col] <= max_lat) & 
            (self.station_info[lon_col] >= min_lon) & 
            (self.station_info[lon_col] <= max_lon)
        ]
        
        # 获取站点ID列名
        id_col = None
        id_candidates = ['id', 'ID', 'station_id', 'stationId', '站点ID', '站点编号', '站点代码', '站号', '站点']
        
        for col in self.station_info.columns:
            if col in id_candidates:
                id_col = col
                break
        
        if id_col is None:
            # 尝试使用第一列作为ID列
            id_col = self.station_info.columns[0]
        
        # 提取站点ID
        station_ids = filtered[id_col].astype(str).tolist()
        logger.info(f"根据经纬度范围 ({min_lat}, {min_lon}) - ({max_lat}, {max_lon}) 筛选出 {len(station_ids)} 个站点")
        
        return station_ids
    
    def _get_stations_from_names(self, station_names):
        """
        根据站点名称获取站点ID列表
        
        参数:
            station_names: 站点名称列表
        
        返回:
            站点ID列表
        """
        if self.station_info is None:
            self._load_station_info()
        
        if self.station_info.empty:
            logger.warning("站点信息为空，无法根据名称查找站点ID")
            return []
        
        # 查找站点名称列
        name_col = None
        name_candidates = ['name', 'Name', 'station_name', 'stationName', '站点名称', '站名', '名称']
        
        for col in self.station_info.columns:
            if col in name_candidates:
                name_col = col
                break
        
        if name_col is None:
            logger.warning("站点信息表中没有找到站点名称列，无法根据名称查找站点ID")
            return []
        
        # 获取站点ID列名
        id_col = None
        id_candidates = ['id', 'ID', 'station_id', 'stationId', '站点ID', '站点编号', '站点代码', '站号', '站点']
        
        for col in self.station_info.columns:
            if col in id_candidates:
                id_col = col
                break
        
        if id_col is None:
            # 尝试使用第一列作为ID列
            id_col = self.station_info.columns[0]
        
        # 根据站点名称查找站点ID
        station_ids = []
        for name in station_names:
            matches = self.station_info[self.station_info[name_col] == name]
            if not matches.empty:
                station_id = str(matches[id_col].iloc[0])
                station_ids.append(station_id)
                logger.info(f"站点名称 '{name}' 匹配到站点ID: {station_id}")
            else:
                logger.warning(f"找不到名为 '{name}' 的站点")
        
        return station_ids
    
    def _get_data_files(self, start_date, end_date):
        """
        获取指定日期范围内的数据文件
        
        参数:
            start_date: 起始日期（字符串格式：'YYYY-MM-DD'）
            end_date: 结束日期（字符串格式：'YYYY-MM-DD'）
            
        返回:
            文件路径列表
        """
        # 将日期字符串转换为日期对象
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # 生成日期范围内的所有日期
        date_range = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_range.append(current_dt)
            current_dt = current_dt + datetime.timedelta(days=1)
        
        # 构建文件名模式
        file_patterns = []
        for dt in date_range:
            date_str = dt.strftime('%Y%m%d')
            file_pattern = f"china_sites_{date_str}.csv"
            file_patterns.append(str(self.data_dir / file_pattern))
        
        # 查找匹配的文件
        matched_files = []
        for pattern in file_patterns:
            for file_path in glob.glob(pattern):
                matched_files.append(file_path)
        
        # 按日期排序
        matched_files.sort()
        
        logger.info(f"找到 {len(matched_files)} 个数据文件")
        return matched_files
    
    def _read_csv_data(self, file_path):
        """
        读取CSV文件数据
        
        参数:
            file_path: CSV文件路径
            
        返回:
            DataFrame对象
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 确保日期列是日期类型
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            
            logger.info(f"成功读取文件 {file_path}，共 {len(df)} 行")
            return df
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时出错: {e}")
            return pd.DataFrame()
    
    def _process_chemical_data(self, data_df, type_name, selected_stations=None):
        """
        处理化学数据，转换为时间序列
        
        参数:
            data_df: 数据DataFrame
            type_name: 污染物类型名称
            selected_stations: 选定的站点列表
            
        返回:
            处理后的DataFrame，以时间为索引，站点为列
        """
        if data_df.empty:
            return pd.DataFrame()
        
        # 筛选指定类型的数据
        type_df = data_df[data_df['type'] == type_name].copy()
        
        if type_df.empty:
            logger.warning(f"没有找到类型为 {type_name} 的数据")
            return pd.DataFrame()
        
        # 创建日期时间列
        type_df['datetime'] = pd.to_datetime(type_df['date']) + pd.to_timedelta(type_df['hour'], unit='h')
        type_df.set_index('datetime', inplace=True)
        
        # 只保留站点列
        columns_to_keep = [col for col in type_df.columns if col not in ['date', 'hour', 'type']]
        
        # 如果提供了站点列表，则只保留这些站点
        if selected_stations:
            available_stations = [col for col in columns_to_keep if col in selected_stations]
            if not available_stations:
                logger.warning(f"选定的站点在数据中不存在")
                return pd.DataFrame()
            
            columns_to_keep = available_stations
        
        # 提取数据
        result_df = type_df[columns_to_keep].copy()
        
        # 将站点数据转换为数值类型
        for col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        
        return result_df
    
    def extract_data(self, start_date, end_date, station_ids=None, station_names=None,
                    min_lat=None, max_lat=None, min_lon=None, max_lon=None, filter_mode='coords'):
        """
        提取指定时间和空间范围内的化学数据
        
        参数:
            start_date: 起始日期，格式：'YYYY-MM-DD'
            end_date: 结束日期，格式：'YYYY-MM-DD'
            station_ids: 站点ID列表
            station_names: 站点名称列表
            min_lat: 最小纬度
            max_lat: 最大纬度
            min_lon: 最小经度
            max_lon: 最大经度
            filter_mode: 筛选模式，'coords'（使用经纬度）或'stations'（使用站点列表）
            
        返回:
            提取的数据字典，键为污染物类型，值为对应的DataFrame
        """
        # 确定要提取的站点
        selected_stations = None
        
        if filter_mode == 'stations' and (station_ids or station_names):
            # 使用站点列表筛选
            if station_ids:
                selected_stations = station_ids
                logger.info(f"使用站点ID列表，共 {len(selected_stations)} 个站点")
            elif station_names:
                selected_stations = self._get_stations_from_names(station_names)
                logger.info(f"从 {len(station_names)} 个站点名称中解析出 {len(selected_stations)} 个站点ID")
        elif filter_mode == 'coords' and min_lat is not None and max_lat is not None and min_lon is not None and max_lon is not None:
            # 使用经纬度范围筛选
            selected_stations = self._filter_stations_by_coords(min_lat, max_lat, min_lon, max_lon)
            logger.info(f"根据经纬度范围筛选出 {len(selected_stations)} 个站点")
            
        # 获取数据文件
        data_files = self._get_data_files(start_date, end_date)
        
        if not data_files:
            logger.error(f"没有找到 {start_date} 到 {end_date} 范围内的数据文件")
            return {}
        
        # 读取和合并所有数据文件
        all_data_df = pd.DataFrame()
        
        for file_path in data_files:
            df = self._read_csv_data(file_path)
            if not df.empty:
                all_data_df = pd.concat([all_data_df, df], ignore_index=True)
        
        if all_data_df.empty:
            logger.error("所有数据文件为空或读取失败")
            return {}
        
        # 获取所有可用的污染物类型
        all_types = all_data_df['type'].unique().tolist()
        logger.info(f"检测到 {len(all_types)} 种污染物类型: {', '.join(all_types)}")
        
        # 处理每种污染物类型
        result_dict = {}
        
        for type_name in all_types:
            logger.info(f"处理 {type_name} 类型的数据...")
            type_df = self._process_chemical_data(all_data_df, type_name, selected_stations)
            
            if not type_df.empty:
                result_dict[type_name] = type_df
                logger.info(f"成功处理 {type_name} 类型的数据，共 {len(type_df)} 个时间点，{len(type_df.columns)} 个站点")
        
        return result_dict
    
    def save_to_excel(self, data_dict):
        """
        将数据保存为Excel文件，每种污染物类型一个工作表
        
        参数:
            data_dict: 数据字典，键为污染物类型，值为对应的DataFrame
            
        返回:
            是否保存成功
        """
        if not data_dict:
            logger.error("没有数据可保存")
            return False
        
        try:
            # 创建Excel写入器
            with pd.ExcelWriter(self.output_file) as writer:
                for type_name, df in data_dict.items():
                    if not df.empty:
                        df.to_excel(writer, sheet_name=type_name)
                        logger.info(f"成功将 {type_name} 类型的数据写入工作表")
            
            logger.info(f"数据已成功保存到 {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存数据到Excel时出错: {e}")
            return False


def main():
    """主函数"""
    start_time = time.time()
    
    logger.info("=== 化学污染物数据提取程序开始运行 ===")
    logger.info(f"数据目录: {DATA_DIR}")
    logger.info(f"输出文件: {OUTPUT_FILE}")
    logger.info(f"时间范围: {START_DATE} 到 {END_DATE}")
    
    # 创建提取器
    extractor = ChemExtractor(DATA_DIR, OUTPUT_FILE, SITE_LOCATION_FILE)
    
    # 确定筛选模式和参数
    if FILTER_MODE == 'stations' and (STATION_IDS or STATION_NAMES):
        logger.info(f"使用站点列表筛选数据")
        if STATION_IDS:
            logger.info(f"站点ID列表: {STATION_IDS}")
        elif STATION_NAMES:
            logger.info(f"站点名称列表: {STATION_NAMES}")
        
        # 提取数据
        data_dict = extractor.extract_data(
            start_date=START_DATE,
            end_date=END_DATE,
            station_ids=STATION_IDS,
            station_names=STATION_NAMES,
            filter_mode='stations'
        )
    else:
        logger.info(f"使用经纬度范围筛选数据: ({MIN_LAT}, {MIN_LON}) - ({MAX_LAT}, {MAX_LON})")
        
        # 提取数据
        data_dict = extractor.extract_data(
            start_date=START_DATE,
            end_date=END_DATE,
            min_lat=MIN_LAT,
            max_lat=MAX_LAT,
            min_lon=MIN_LON,
            max_lon=MAX_LON,
            filter_mode='coords'
        )
    
    # 保存数据
    if data_dict:
        success = extractor.save_to_excel(data_dict)
        if success:
            logger.info("数据提取和保存成功")
        else:
            logger.error("数据保存失败")
    else:
        logger.error("无数据可保存")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"程序运行完毕，耗时: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    main()
