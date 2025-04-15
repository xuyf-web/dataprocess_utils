#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-

# ============================================================
# 参数配置区域（根据需要修改以下参数）
# ============================================================

# 输入/输出参数
DATA_DIR = '/data8/xuyf/data/obs/meteo_gsod/data_2025'  # GSOD数据所在目录
OUTPUT_FILE = 'gsod_2025.xlsx'    # 输出Excel文件路径
TEMP_DIR = None                   # 临时目录，用于解压数据（None表示使用系统临时目录）

# 时间范围参数（以下三个参数选择一个设置，优先级：YEAR > START_DATE/END_DATE）
YEAR = 2025                       # 要提取的年份，例如：2025
START_DATE = '2025-01-01'                 # 起始日期，格式：'YYYY-MM-DD'，例如：'2025-01-01'
END_DATE = '2025-01-03'                   # 结束日期，格式：'YYYY-MM-DD'，例如：'2025-12-31'

# 空间范围选择参数
# 当同时设置了经纬度范围和站点列表时，此参数决定使用哪种方式过滤数据
# 可选值：'coords'（使用经纬度范围）或 'stations'（使用站点列表）
FILTER_MODE = 'coords'            # 空间范围过滤模式

# 经纬度范围参数
MIN_LAT = 30                    # 最小纬度，例如：35.0
MAX_LAT = 40                    # 最大纬度，例如：40.0
MIN_LON = 115                    # 最小经度，例如：115.0
MAX_LON = 120                    # 最大经度，例如：120.0

# 站点列表参数
STATION_NAMES = None              # 站点名称列表，例如：['BEIJING', 'SHANGHAI']
STATION_IDS = None                # 站点ID列表，例如：['95807099999', '95818099999']

# 数据处理参数
CLEAN_DATA = True                 # 是否进行数据清洗（True/False）

# ============================================================
# 以下是脚本代码，一般不需要修改
# ============================================================

import os
import tarfile
import pandas as pd
import glob
import numpy as np
from pathlib import Path
import shutil
import tempfile
import datetime

class GSODExtractor:
    """
    全球气象观测数据提取器
    用于按照时间范围和空间范围提取GSOD气象数据并保存为Excel文件
    """
    
    def __init__(self, data_dir, output_file, temp_dir=None):
        """
        初始化提取器
        
        参数:
            data_dir: 气象数据所在目录
            output_file: 输出Excel文件路径
            temp_dir: 临时目录，如果不指定则使用系统临时目录
        """
        self.data_dir = Path(data_dir)
        self.output_file = output_file
        self.temp_dir = temp_dir
        self.temp_extract_dir = None
        self.station_info = None
        
    def _extract_tar(self, tar_file, year):
        """解压tar文件到临时目录"""
        # 创建临时目录
        if self.temp_dir:
            extract_dir = Path(self.temp_dir) / f"gsod_{year}"
            os.makedirs(extract_dir, exist_ok=True)
            self.temp_extract_dir = extract_dir
        else:
            self.temp_extract_dir = Path(tempfile.mkdtemp(prefix=f"gsod_{year}_"))
            
        print(f"解压数据到临时目录: {self.temp_extract_dir}")
        
        # 解压文件
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(path=self.temp_extract_dir)
            
        print(f"解压完成")
        return self.temp_extract_dir
    
    def _load_station_info(self):
        """加载站点信息"""
        # 查找站点信息文件
        isd_history_file = None
        for ext in ['.csv', '.txt']:
            potential_file = self.data_dir / "doc" / f"isd-history{ext}"
            if potential_file.exists():
                isd_history_file = potential_file
                break
                
        if not isd_history_file:
            print("警告: 找不到站点信息文件，将无法按站点名称过滤")
            self.station_info = pd.DataFrame(columns=['USAF', 'WBAN', 'STATION NAME', 'LAT', 'LON'])
            return
        
        # 加载站点信息
        try:
            if isd_history_file.suffix == '.csv':
                self.station_info = pd.read_csv(isd_history_file)
            else:
                # 尝试加载txt格式的站点信息
                try:
                    self.station_info = pd.read_csv(isd_history_file, delimiter=',')
                except:
                    self.station_info = pd.read_fwf(isd_history_file)
            
            # 确保列名标准化
            if 'STATION NAME' not in self.station_info.columns and 'STATION' in self.station_info.columns:
                self.station_info = self.station_info.rename(columns={'STATION': 'STATION NAME'})
            if 'LAT' not in self.station_info.columns and 'LATITUDE' in self.station_info.columns:
                self.station_info = self.station_info.rename(columns={'LATITUDE': 'LAT'})
            if 'LON' not in self.station_info.columns and 'LONGITUDE' in self.station_info.columns:
                self.station_info = self.station_info.rename(columns={'LONGITUDE': 'LON'})
            
            print(f"成功加载了 {len(self.station_info)} 个站点信息")
        except Exception as e:
            print(f"加载站点信息时出错: {e}")
            self.station_info = pd.DataFrame(columns=['USAF', 'WBAN', 'STATION NAME', 'LAT', 'LON'])
    
    def _get_station_from_name(self, station_names):
        """根据站点名称获取站点ID"""
        if self.station_info is None:
            self._load_station_info()
            
        station_ids = []
        for name in station_names:
            # 模糊匹配站点名称
            matches = self.station_info[self.station_info['STATION NAME'].str.contains(name, case=False, na=False)]
            if not matches.empty:
                for _, row in matches.iterrows():
                    station_id = str(row['USAF']).zfill(6) + str(row.get('WBAN', '99999')).zfill(5)
                    station_ids.append(station_id)
                    print(f"站点名称 '{name}' 匹配到站点 {row['STATION NAME']} (ID: {station_id})")
            else:
                print(f"警告: 找不到名称包含 '{name}' 的站点")
                
        return station_ids
    
    def _filter_by_coords(self, df, min_lat, max_lat, min_lon, max_lon):
        """按经纬度范围过滤数据"""
        # 确保经纬度列是数值型
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
        
        # 按经纬度过滤
        mask = (
            (df['LATITUDE'] >= min_lat) & 
            (df['LATITUDE'] <= max_lat) & 
            (df['LONGITUDE'] >= min_lon) & 
            (df['LONGITUDE'] <= max_lon)
        )
        return df[mask]
    
    def _process_csv_files(self, csv_files, start_date, end_date, stations=None, 
                          min_lat=None, max_lat=None, min_lon=None, max_lon=None,
                          filter_mode='coords'):
        """处理CSV文件并按条件过滤数据"""
        all_data = []
        
        # 将日期字符串转换为datetime对象
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # 确定是否执行经纬度过滤和站点过滤
        do_coord_filter = (min_lat is not None and max_lat is not None and 
                          min_lon is not None and max_lon is not None)
        do_station_filter = (stations is not None and len(stations) > 0)

        # 如果两种过滤条件都满足，根据filter_mode决定使用哪种
        use_coords = False
        use_stations = False
        
        if do_coord_filter and do_station_filter:
            # 两种都具备，根据过滤模式选择
            if filter_mode.lower() == 'coords':
                use_coords = True
                print("同时设置了经纬度范围和站点列表，根据FILTER_MODE设置，将使用经纬度范围过滤数据")
            else:  # 'stations'
                use_stations = True
                print("同时设置了经纬度范围和站点列表，根据FILTER_MODE设置，将使用站点列表过滤数据")
        else:
            # 只有一种过滤条件
            use_coords = do_coord_filter
            use_stations = do_station_filter
            
        for csv_file in csv_files:
            try:
                # 从文件名中提取站点ID
                station_id = Path(csv_file).stem
                
                # 如果使用站点过滤且当前站点不在列表中，则跳过
                if use_stations and station_id not in stations:
                    continue
                
                # 读取CSV文件
                df = pd.read_csv(csv_file)
                
                # 确保DATE列是日期类型
                df['DATE'] = pd.to_datetime(df['DATE'])
                
                # 按日期范围过滤
                df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
                
                # 如果在日期范围内没有数据，则跳过
                if df.empty:
                    continue
                
                # 如果使用经纬度过滤，则按经纬度过滤
                if use_coords:
                    df = self._filter_by_coords(df, min_lat, max_lat, min_lon, max_lon)
                    
                    # 如果过滤后没有数据，则跳过
                    if df.empty:
                        continue
                
                # 添加到结果集
                all_data.append(df)
                
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                
        # 合并所有数据
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            print(f"总共提取了 {len(result)} 条记录")
            return result
        else:
            print("没有找到符合条件的数据")
            return pd.DataFrame()
    
    def _clean_data(self, df):
        """清洗数据，处理缺失值和单位转换"""
        # 复制DataFrame以避免修改原始数据
        cleaned = df.copy()
        
        # 处理温度数据 (华氏度转摄氏度)
        for col in ['TEMP', 'DEWP', 'MAX', 'MIN']:
            if col in cleaned.columns:
                # 转换为数值类型
                cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
                # 缺失值处理 (9999.9 表示缺失)
                cleaned.loc[cleaned[col] > 9000, col] = np.nan
                # 华氏度转摄氏度
                cleaned[col] = (cleaned[col] - 32) * 5/9
        
        # 处理压力数据 (毫巴转百帕)
        for col in ['SLP', 'STP']:
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
                cleaned.loc[cleaned[col] > 9000, col] = np.nan
                # 毫巴转百帕 (1毫巴 = 1百帕)
                cleaned[col] = cleaned[col] * 1.0
        
        # 处理风速数据 (节转换为米/秒)
        for col in ['WDSP', 'MXSPD', 'GUST']:
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
                cleaned.loc[cleaned[col] > 900, col] = np.nan
                # 节转米/秒 (1节 = 0.514444米/秒)
                cleaned[col] = cleaned[col] * 0.514444
        
        # 处理降水量数据 (英寸转毫米)
        if 'PRCP' in cleaned.columns:
            cleaned['PRCP'] = pd.to_numeric(cleaned['PRCP'], errors='coerce')
            cleaned.loc[cleaned['PRCP'] > 90, 'PRCP'] = np.nan
            # 英寸转毫米 (1英寸 = 25.4毫米)
            cleaned['PRCP'] = cleaned['PRCP'] * 25.4
        
        # 处理能见度数据 (英里转千米)
        if 'VISIB' in cleaned.columns:
            cleaned['VISIB'] = pd.to_numeric(cleaned['VISIB'], errors='coerce')
            cleaned.loc[cleaned['VISIB'] > 900, 'VISIB'] = np.nan
            # 英里转千米 (1英里 = 1.60934千米)
            cleaned['VISIB'] = cleaned['VISIB'] * 1.60934
        
        # 处理积雪深度数据 (英寸转厘米)
        if 'SNDP' in cleaned.columns:
            cleaned['SNDP'] = pd.to_numeric(cleaned['SNDP'], errors='coerce')
            cleaned.loc[cleaned['SNDP'] > 900, 'SNDP'] = np.nan
            # 英寸转厘米 (1英寸 = 2.54厘米)
            cleaned['SNDP'] = cleaned['SNDP'] * 2.54
            
        # 重命名列以反映单位变化
        column_mapping = {
            'TEMP': 'TEMP_C',          # 平均温度（摄氏度）
            'DEWP': 'DEWP_C',          # 平均露点温度（摄氏度）
            'SLP': 'SLP_HPA',          # 平均海平面气压（百帕）
            'STP': 'STP_HPA',          # 平均站点气压（百帕）
            'VISIB': 'VISIB_KM',       # 平均能见度（千米）
            'WDSP': 'WDSP_MS',         # 平均风速（米/秒）
            'MXSPD': 'MXSPD_MS',       # 最大持续风速（米/秒）
            'GUST': 'GUST_MS',         # 最大阵风（米/秒）
            'MAX': 'MAX_C',            # 最高温度（摄氏度）
            'MIN': 'MIN_C',            # 最低温度（摄氏度）
            'PRCP': 'PRCP_MM',         # 降水量（毫米）
            'SNDP': 'SNDP_CM'          # 积雪深度（厘米）
        }
        cleaned = cleaned.rename(columns=column_mapping)
        
        # 添加气象状况的易读标签
        if 'FRSHTT' in cleaned.columns:
            # 确保FRSHTT是字符串类型
            cleaned['FRSHTT'] = cleaned['FRSHTT'].astype(str)
            # 确保FRSHTT是6位字符串，不足的补0
            cleaned['FRSHTT'] = cleaned['FRSHTT'].str.zfill(6)
            cleaned['FOG'] = cleaned['FRSHTT'].str[0].astype(int)
            cleaned['RAIN'] = cleaned['FRSHTT'].str[1].astype(int)
            cleaned['SNOW'] = cleaned['FRSHTT'].str[2].astype(int)
            cleaned['HAIL'] = cleaned['FRSHTT'].str[3].astype(int)
            cleaned['THUNDER'] = cleaned['FRSHTT'].str[4].astype(int)
            cleaned['TORNADO'] = cleaned['FRSHTT'].str[5].astype(int)
        
        return cleaned
    
    def _get_station_info_df(self, df):
        """获取站点信息DataFrame"""
        # 获取唯一的站点信息
        station_info = df[['STATION', 'NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
        # 处理站点名称，分离站点名和国别
        station_info['STATION_NAME'] = station_info['NAME'].apply(lambda x: x.split(',')[0] if pd.notna(x) and ',' in x else x)
        station_info['COUNTRY'] = station_info['NAME'].apply(lambda x: x.split(',')[1].strip() if pd.notna(x) and ',' in x else '')
        return station_info

    def _get_param_info_df(self):
        """获取参数说明DataFrame"""
        param_info = pd.DataFrame({
            '参数名': [
                'STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 'ELEVATION',
                'TEMP_C', 'TEMP_ATTRIBUTES', 'DEWP_C', 'DEWP_ATTRIBUTES',
                'SLP_HPA', 'SLP_ATTRIBUTES', 'STP_HPA', 'STP_ATTRIBUTES',
                'VISIB_KM', 'VISIB_ATTRIBUTES', 'WDSP_MS', 'WDSP_ATTRIBUTES',
                'MXSPD_MS', 'MXSPD_ATTRIBUTES', 'GUST_MS', 'GUST_ATTRIBUTES',
                'MAX_C', 'MAX_ATTRIBUTES', 'MIN_C', 'MIN_ATTRIBUTES',
                'PRCP_MM', 'PRCP_ATTRIBUTES', 'SNDP_CM', 'SNDP_ATTRIBUTES',
                'FOG', 'RAIN', 'SNOW', 'HAIL', 'THUNDER', 'TORNADO'
            ],
            '含义': [
                '站点ID', '日期', '纬度', '经度', '海拔高度',
                '平均温度', '用于计算平均温度的观测次数', '平均露点温度', '用于计算平均露点温度的观测次数',
                '平均海平面气压', '用于计算平均海平面气压的观测次数', '平均站点气压', '用于计算平均站点气压的观测次数',
                '平均能见度', '用于计算平均能见度的观测次数', '平均风速', '用于计算平均风速的观测次数',
                '最大持续风速', '用于计算最大持续风速的观测次数', '最大阵风', '用于计算最大阵风的观测次数',
                '最高温度', '最高温度来源：空白=直接报告，*=从小时数据推导', '最低温度', '最低温度来源：空白=直接报告，*=从小时数据推导',
                '降水量', '降水量数据来源：A=1个6小时报告，B=2个6小时报告，C=3个6小时报告，D=4个6小时报告，E=1个12小时报告，F=2个12小时报告，G=1个24小时报告，H=报告为0但可能有微量降水，I=无降水报告', '积雪深度', '积雪深度观测次数',
                '是否有雾', '是否有雨', '是否有雪', '是否有冰雹', '是否有雷暴', '是否有龙卷风'
            ],
            '单位': [
                '无', '无', '度', '度', '米',
                '摄氏度', '次', '摄氏度', '次',
                '百帕', '次', '百帕', '次',
                '千米', '次', '米/秒', '次',
                '米/秒', '次', '米/秒', '次',
                '摄氏度', '无', '摄氏度', '无',
                '毫米', '无', '厘米', '次',
                '0/1', '0/1', '0/1', '0/1', '0/1', '0/1'
            ]
        })
        return param_info

    def save_to_excel(self, df, sheet_name='气象数据'):
        """将数据保存为Excel文件，按站点分组保存到不同sheet"""
        if df.empty:
            print("没有数据可保存")
            return False
        
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
            
            # 保存为Excel
            print(f"正在保存数据到 {self.output_file}")
            with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                # 保存参数说明sheet
                param_info = self._get_param_info_df()
                param_info.to_excel(writer, sheet_name='param', index=False)
                
                # 保存站点信息sheet
                station_info = self._get_station_info_df(df)
                # 重新排列station sheet的列顺序
                station_info = station_info[['STATION', 'STATION_NAME', 'COUNTRY', 'LATITUDE', 'LONGITUDE']]
                station_info.to_excel(writer, sheet_name='station', index=False)
                
                # 按站点分组保存数据
                for station_id in df['STATION'].unique():
                    station_data = df[df['STATION'] == station_id].copy()
                    # 设置DATE为索引
                    station_data.set_index('DATE', inplace=True)
                    # 获取站点名称
                    station_name = station_data['NAME'].iloc[0]
                    if pd.notna(station_name) and ',' in station_name:
                        station_name = station_name.split(',')[0]
                    # 保存到sheet（限制sheet名称为31个字符）
                    sheet_name = str(station_name)[:31]
                    # 删除不需要的列
                    columns_to_drop = ['NAME', 'STATION', 'LATITUDE', 'LONGITUDE']
                    station_data = station_data.drop(columns=[col for col in columns_to_drop if col in station_data.columns])
                    station_data.to_excel(writer, sheet_name=sheet_name)
                
            print(f"数据已成功保存到 {self.output_file}")
            return True
        except Exception as e:
            print(f"保存数据时出错: {e}")
            return False

    def extract_data(self, year=None, start_date=None, end_date=None, 
                   station_names=None, station_ids=None,
                   min_lat=None, max_lat=None, min_lon=None, max_lon=None,
                   clean_data=True, filter_mode='coords'):
        """
        从GSOD数据中提取指定条件的气象数据
        
        参数:
            year: 要提取的年份，如果未指定则尝试从起止日期推断
            start_date: 起始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            station_names: 站点名称列表
            station_ids: 站点ID列表
            min_lat: 最小纬度
            max_lat: 最大纬度
            min_lon: 最小经度
            max_lon: 最大经度
            clean_data: 是否清洗数据（处理缺失值和单位转换）
            filter_mode: 当同时设置经纬度范围和站点列表时的过滤模式（'coords'或'stations'）
            
        返回:
            pandas.DataFrame: 提取的数据
        """
        # 设置默认日期范围
        if start_date is None and end_date is None:
            if year:
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
            else:
                # 默认提取当前年份的数据
                current_year = datetime.datetime.now().year
                start_date = f"{current_year}-01-01"
                end_date = f"{current_year}-12-31"
                year = current_year
        elif start_date and end_date is None:
            # 只指定了起始日期，提取起始日期至当前的数据
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        elif end_date and start_date is None:
            # 只指定了结束日期，提取结束日期前一年的数据
            end_date_obj = pd.to_datetime(end_date)
            start_date = (end_date_obj - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
            
        # 从日期推断年份范围
        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year
        years = list(range(start_year, end_year + 1))
        
        # 确定站点列表
        stations = []
        if station_ids:
            stations.extend(station_ids)
        if station_names:
            self._load_station_info()
            name_stations = self._get_station_from_name(station_names)
            stations.extend(name_stations)
            
        # 累积所有年份的数据
        all_data = []
        
        # 处理每个年份的数据
        for year in years:
            print(f"处理 {year} 年的数据")
            
            # 首先在DATA_DIR下查找CSV文件
            csv_files = list(self.data_dir.glob("*.csv"))
            
            if csv_files:
                print(f"在 {self.data_dir} 中找到 {len(csv_files)} 个CSV文件")
                # 处理CSV文件
                df = self._process_csv_files(
                    csv_files, start_date, end_date, stations,
                    min_lat, max_lat, min_lon, max_lon,
                    filter_mode=filter_mode
                )
                
                if not df.empty:
                    all_data.append(df)
            else:
                print(f"在 {self.data_dir} 中没有找到CSV文件，尝试查找tar文件")
                # 查找tar.gz文件
                tar_pattern = self.data_dir / f"{year}.tar.gz"
                tar_files = list(glob.glob(str(tar_pattern)))
                
                if tar_files:
                    # 解压tar文件
                    tar_file = tar_files[0]
                    extract_dir = self._extract_tar(tar_file, year)
                    
                    # 获取解压后的CSV文件
                    csv_files = list(extract_dir.glob("*.csv"))
                    
                    if csv_files:
                        print(f"从tar文件中解压出 {len(csv_files)} 个CSV文件")
                        # 处理CSV文件
                        df = self._process_csv_files(
                            csv_files, start_date, end_date, stations,
                            min_lat, max_lat, min_lon, max_lon,
                            filter_mode=filter_mode
                        )
                        
                        if not df.empty:
                            all_data.append(df)
                    else:
                        print(f"警告: 解压后没有找到CSV文件")
                    
                    # 清理临时目录
                    if self.temp_extract_dir and self.temp_dir is None:
                        shutil.rmtree(self.temp_extract_dir)
                        print(f"已清理临时目录 {self.temp_extract_dir}")
                else:
                    print(f"警告: 找不到 {year} 年的数据文件")
                    continue
                
        # 合并所有年份的数据
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            
            # 数据清洗
            if clean_data:
                result = self._clean_data(result)
                
            # 按日期排序
            result = result.sort_values('DATE')
            
            print(f"成功提取了 {len(result)} 条记录")
            return result
        else:
            print("没有找到符合条件的数据")
            return pd.DataFrame()

def main():
    """主函数"""
    # 创建提取器
    extractor = GSODExtractor(
        data_dir=DATA_DIR,
        output_file=OUTPUT_FILE,
        temp_dir=TEMP_DIR
    )
    
    # 提取数据
    df = extractor.extract_data(
        year=YEAR,
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

if __name__ == '__main__':
    main() 