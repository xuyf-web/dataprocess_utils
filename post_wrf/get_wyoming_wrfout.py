#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
'''
This script is used to get vertical profile data from wrfout files at multiple stations in Wyoming format
Author: Xuyf
Date: 2025-04-14
'''

# ============================================================
# User Configuration Options (Modify as needed)
# ============================================================

# Path Settings
WRFOUT_DIR = '/data8/xuyf/Project/Xishan/data/wrfout'
OUTPUT_DIR = '/data8/xuyf/Project/Xishan/data/postwrf/wyoming'
DOMAIN = 'd01'

# Time Range Settings
START_DATE = '2024-07-22 00:00:00'
END_DATE = '2024-07-25 00:00:00'

# Station Locations
STATIONS = {
    # 'STATION_NAME': [LON, LAT]
      'NANJING'     : [118.9, 31.93],
    #   'SHANGHAI'    : [121.45, 31.42],
    #   'HANGZHOU'    : [120.17, 30.23],
    #   'QUXIAN'      : [118.87, 28.97],
    #   'ANQING'      : [116.97, 30.62],
    #   'FUYANG'      : [115.73, 32.87],
    #   'XUZHOU'      : [117.15, 34.28],
    #   'SHEYANG'     : [120.3, 33.75],
}

# Whether to Process Chemical Species
CHEM = True

CHEMICAL_VARS = {
# 'var_name in output': 'var_name in wrfout'
  'O3'                   : 'o3',
  'NO2'                  : 'no2',
  'NO'                   : 'no',
  'CO'                   : 'co',
  'ISOP'                 : 'iso',       # iso for CBMZ, isoprene for SAPRC99
  'PM25'                 : 'PM2_5_DRY',
  'PM10'                 : 'PM10',
  'BC'                   : 'bc',    # bc_a0* for sum up all bc
}

# 日志文件路径
LOG_FILE = "get_wyoming_wrfout.log"

# ============================================================
# Script Code (Do not modify)
# ============================================================

import os
import re
import time
import numpy as np
import xarray as xr
import pandas as pd
import dask
import logging
from findpoint import nearest_position

# Remove log file if exists
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 气象变量定义
# Meteorological variable definitions
met_vars = ['T', 'QVAPOR', 'P', 'PB', 'PH', 'PHB']

# 需要移除的变量和坐标
# Variables and coordinates to remove
rm_coords = ['XLONG', 'XLAT', 'XLONG_U', 'XLAT_U', 'XLONG_V', 'XLAT_V', 'XTIME']

# 配置dask以避免内存问题
dask.config.set({"array.chunk-size": "64MiB"})

def generate_filelist(datapath, startdate, enddate, domain):
    """
    生成WRF输出文件列表（仅00:00和12:00）
    Generate WRF output file list (only 00:00 and 12:00)
    
    Args:
        datapath: WRF输出文件路径 (WRF output file path)
        startdate: 开始时间 (Start time with timezone)
        enddate: 结束时间 (End time with timezone)
        domain: 模拟域 (Simulation domain)
    Returns:
        filelist: 文件列表 (File list)
        timelist: 时间列表 (Time list)
    """
    if not datapath.endswith(os.sep):
        datapath += os.sep

    date_range = pd.date_range(start=startdate, end=enddate, freq='D')
    
    filelist = []
    timelist = []
    file_count = 0
    missing_count = 0
    
    for day in date_range:
        # 当天00时
        local00_to_utc = day.tz_convert('UTC')
        utc_filename_format = local00_to_utc.strftime('%Y-%m-%d_%H:%M:%S')
        filename00 = f"wrfout_{domain}_{utc_filename_format}"
        full_path00 = os.path.join(datapath, filename00)
        
        if os.path.exists(full_path00):
            filelist.append(full_path00)
            timelist.append(local00_to_utc)
            file_count += 1
        else:
            missing_count += 1
            logger.warning(f"File not found: {os.path.basename(full_path00)}")
        
        # 当天12时
        local12_to_utc = (day + pd.Timedelta(hours=12)).tz_convert('UTC')
        utc_filename_format = local12_to_utc.strftime('%Y-%m-%d_%H:%M:%S')
        filename12 = f"wrfout_{domain}_{utc_filename_format}"
        full_path12 = os.path.join(datapath, filename12)
        
        if os.path.exists(full_path12):
            filelist.append(full_path12)
            timelist.append(local12_to_utc)
            file_count += 1
        else:
            missing_count += 1
            logger.warning(f"File not found: {os.path.basename(full_path12)}")
    
    if not filelist:
        logger.error(f"No valid files found in {datapath} for domain {domain}")
        logger.info(f"Looking for files matching pattern: wrfout_{domain}_YYYY-MM-DD_HH:MM:SS")
    else:
        logger.info(f"Found {file_count} files for processing")
        if missing_count > 0:
            logger.warning(f"Missing {missing_count} expected files")
            
    return filelist, timelist

def combine_wrfout(datapath, start, end, domain='d01'):
    """
    合并WRF输出文件
    Combine WRF output files
    
    Args:
        datapath: WRF输出文件路径 (WRF output file path)
        start: 开始时间 (Start time)
        end: 结束时间 (End time)
        domain: 模拟域 (Simulation domain)
    Returns:
        dataset: 合并后的数据集 (Combined dataset)
        timelist: 时间列表 (Time list for wyoming mode)
    """
    logger.info(f"Reading data from {datapath}")
    
    # 调整UTC到本地时间
    startdate = pd.Timestamp(start).tz_localize('Asia/Shanghai')
    enddate = pd.Timestamp(end).tz_localize('Asia/Shanghai')
    
    filelist, timelist = generate_filelist(datapath, startdate, enddate, domain)
    
    dataset = xr.open_mfdataset(filelist, combine='nested', concat_dim='Time')
    
    # 调整时间坐标，使用00:00和12:00的时间
    local_times = [t.tz_convert('Asia/Shanghai').tz_localize(None) for t in timelist]
    dataset = dataset.assign_coords(Time=('Time', local_times))
    logger.info("Time coordinates adjusted to local time (00:00 and 12:00 only)")
    
    return dataset


def calculate_temperature(dataset):
    """
    计算温度
    Calculate temperature
    """
    theta = dataset['T']  # 位温扰动 (potential temperature perturbation)
    p = dataset['P'] + dataset['PB']  # 压力, Pa (pressure, Pa)
    t = (theta + 300) * (p / 100000) ** (2/7) - 273.15  # 温度, C (temperature, C)
    return t

def calculate_relative_humidity(dataset):
    """
    计算相对湿度
    Calculate relative humidity
    """
    t = calculate_temperature(dataset)
    q = dataset['QVAPOR']  # 水汽混合比, kg/kg (water vapor mixing ratio, kg/kg)
    p = (dataset['P'] + dataset['PB']) / 100  # 压力, hPa (pressure, hPa)
    e = p * q / (0.622 + q)
    es = 6.112 * np.exp(17.67 * t / (t + 243.5))
    rh = e / es * 100
    return rh

def detect_available_chemical_vars(ds):
    """
    检测数据集中可用的化学变量
    Detect available chemical variables in the dataset
    """
    available_vars = {}
    found_count = 0

    for output_var, wrfout_var in CHEMICAL_VARS.items():
        if wrfout_var == 'bc':
            bc_vars = [key for key in ds.keys() if re.match(r'bc_a\d+', key)]
            logger.info(f"Found {len(bc_vars)} bc_a* variables -> {output_var}")
            found_count += len(bc_vars)
        elif wrfout_var in ds:
            found_count += 1
            available_vars[output_var] = wrfout_var
            logger.info(f"Found variable: {wrfout_var} -> {output_var}")
        else:
            logger.warning(f"Cannot find variable: {wrfout_var}")
    return available_vars

def process_wrf_wyoming(datapath):
    """
    处理WRF多站点数据（wyoming模式）
    Process WRF multi-station data (wyoming mode)
    """
    # 合并WRF输出文件
    ds = combine_wrfout(datapath, START_DATE, END_DATE, DOMAIN)
    
    # 检测可用的化学变量
    available_chem_vars = detect_available_chemical_vars(ds) if CHEM else {}
    
    # 获取经纬度网格数据
    lon = ds['XLONG'][0, :, :].squeeze()
    lat = ds['XLAT'][0, :, :].squeeze()
    lon_u = ds['XLONG_U'][0, :, :].squeeze()
    lat_u = ds['XLAT_U'][0, :, :].squeeze()
    lon_v = ds['XLONG_V'][0, :, :].squeeze()
    lat_v = ds['XLAT_V'][0, :, :].squeeze()
    
    # 存储各站点数据集
    station_datasets = {}
    
    # 循环处理每个站点
    for station_name, (std_lon, std_lat) in STATIONS.items():
        logger.info(f"Processing station: {station_name}")
        
        # 查找最近网格点
        nr_point = nearest_position(std_lon, std_lat, lon.values, lat.values)
        
        # 提取基础变量
        logger.info("Processing meteorological variables")
        dataset = ds[met_vars].isel(south_north=nr_point[0], west_east=nr_point[1])
        
        # 计算衍生变量
        dataset['Temp'] = calculate_temperature(dataset)
        dataset['RH'] = calculate_relative_humidity(dataset)
        dataset['Pres'] = (dataset['P'] + dataset['PB']) / 100  # 转换为hPa
        dataset['Height'] = (dataset['PH'] + dataset['PHB']) / 9.8
        
        # 处理U/V变量（注意交错网格）
        nr_point_u = nearest_position(std_lon, std_lat, lon_u.values, lat_u.values)
        nr_point_v = nearest_position(std_lon, std_lat, lon_v.values, lat_v.values)
        dataset['U'] = ds['U'].isel(south_north=nr_point_u[0], west_east_stag=nr_point_u[1]).squeeze()
        dataset['V'] = ds['V'].isel(south_north_stag=nr_point_v[0], west_east=nr_point_v[1]).squeeze()
        dataset['WS'] = (dataset['U']**2 + dataset['V']**2)**0.5
        dataset['WD'] = np.degrees(np.arctan2(-dataset['U'], -dataset['V'])) % 360
        
        dataset = dataset.drop_vars(met_vars, errors='ignore')
        
        if CHEM:
            logger.info("Processing chemical variables")
            # 处理化学变量
            for output_var, wrfout_var in available_chem_vars.items():
                dataset[output_var] = ds[wrfout_var].isel(south_north=nr_point[0], west_east=nr_point[1])
            
            # 黑碳处理
            if 'bc' in CHEMICAL_VARS.values():
                bc_vars = [key for key in ds.keys() if re.match(r'bc_a\d+', key)]
                if bc_vars:
                    dataset['BC'] = sum(ds[key].isel(south_north=nr_point[0], west_east=nr_point[1]) for key in bc_vars)
                    logger.info(f"Processed black carbon from {len(bc_vars)} variables")
                else:
                    logger.warning("Cannot find black carbon variables (bc_a0*)")
        
        station_datasets[station_name] = dataset.drop_vars(rm_coords, errors='ignore')
    
    logger.info("Processing of wrfout data completed.")
    return station_datasets

def convert_to_excel(station_datasets, output_dir):
    """
    将数据集转换为Excel文件
    Convert datasets to Excel files
    """
    output_vars = ['Temp', 'RH', 'Pres', 'U', 'V', 'WS', 'WD', 'Height']
    if CHEM:
        output_vars.extend(list(CHEMICAL_VARS.keys()))
    
    stations_processed = 0
    
    for station_name, ds in station_datasets.items():
        output_file = os.path.join(output_dir, f"{station_name}.xlsx")
        logger.info(f"Converting to Excel: {output_file}")
        
        with pd.ExcelWriter(output_file) as writer:
            vars_missing = 0
            for var in output_vars:
                if var in ds:
                    # 转换为DataFrame并处理多级索引
                    df = ds[var].to_dataframe().unstack(level=1)
                    df.columns = df.columns.droplevel(0)
                    
                    # 确保索引没有时区信息
                    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    df.to_excel(writer, sheet_name=var)
                    logger.info(f"Exported {var}")
                else:
                    vars_missing += 1
                    logger.warning(f"Variable '{var}' not in dataset for station {station_name}")
        
        if vars_missing > 0:
            logger.warning(f"{vars_missing} variables were not available for station {station_name}")
        
        stations_processed += 1
    
    return stations_processed

def main():
    prog_start_time = time.time()
    
    logger.info(f"Processing Wyoming format data from {START_DATE} to {END_DATE}")
    logger.info(f"Processing {len(STATIONS)} stations in domain {DOMAIN}")
    
    # 处理WRF多站点剖面数据
    station_datasets = process_wrf_wyoming(WRFOUT_DIR)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 导出数据
    stations_processed = convert_to_excel(station_datasets, OUTPUT_DIR)
    
    prog_end_time = time.time()
    elapsed_time = prog_end_time - prog_start_time
    
    logger.info(f"Processed {stations_processed} stations")
    if elapsed_time < 60:
        logger.info(f"Completed in {elapsed_time:.2f} seconds")
    elif elapsed_time < 3600:
        logger.info(f"Completed in {elapsed_time / 60:.2f} minutes")
    else:
        logger.info(f"Completed in {elapsed_time / 3600:.2f} hours")
    
if __name__ == "__main__":
    main() 