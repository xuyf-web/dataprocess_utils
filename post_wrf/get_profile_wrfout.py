#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
'''
This script is used to get vertical profile data from wrfout files at a single location
Author: Xuyf
Date: 2025-04-14
'''

# ============================================================
# User Configuration Options (Modify as needed)
# ============================================================

# Path Settings
WRFOUT_DIR = '/data8/xuyf/Project/Xishan/data/wrfout'
OUTPUT_DIR = '/data8/xuyf/Project/Xishan/data/postwrf'
DOMAIN = 'd01'
OUTPUT_FILENAME = 'profile.xlsx'

# Time Range Settings
START_DATE = '2024-07-22 00:00:00'
END_DATE = '2024-09-08 00:00:00'

# Location Settings
LONGITUDE = 116.72
LATITUDE = 32.59

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
LOG_FILE = "get_profile_wrfout.log"

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
    生成WRF输出文件列表
    Generate WRF output file list
    
    Args:
        datapath: WRF输出文件路径 (WRF output file path)
        startdate: 开始时间 (Start time with timezone)
        enddate: 结束时间 (End time with timezone)
        domain: 模拟域 (Simulation domain)
    Returns:
        filelist: 文件列表 (File list)
    """
    if not datapath.endswith(os.sep):
        datapath += os.sep

    startutc = startdate.tz_convert('UTC')
    endutc = enddate.tz_convert('UTC')
    logger.info(f"UTC time from {startutc.strftime('%Y-%m-%d %H:%M')} to {endutc.strftime('%Y-%m-%d %H:%M')}")
    date_range = pd.date_range(start=startutc, end=endutc, freq='h')
    
    filelist = []
    file_count = 0
    missing_count = 0
    
    for d in date_range:
        utc_filename_format = d.strftime('%Y-%m-%d_%H:%M:%S')
        filename = f"wrfout_{domain}_{utc_filename_format}"
        full_path = os.path.join(datapath, filename)
        if os.path.exists(full_path):
            filelist.append(full_path)
            file_count += 1
        else:
            missing_count += 1
            
    if not filelist:
        logger.error(f"No valid files found in {datapath} for domain {domain}")
        logger.info(f"Looking for files matching pattern: wrfout_{domain}_YYYY-MM-DD_HH:MM:SS")
    else:
        logger.info(f"Found {file_count} files for processing")
        if missing_count > 0:
            logger.warning(f"Missing {missing_count} expected files")
            
    return filelist

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
    """
    logger.info(f"Reading data from {datapath}")
    
    # 调整UTC到本地时间
    startdate = pd.Timestamp(start).tz_localize('Asia/Shanghai')
    enddate = pd.Timestamp(end).tz_localize('Asia/Shanghai')
    
    filelist = generate_filelist(datapath, startdate, enddate, domain)
    
    dataset = xr.open_mfdataset(filelist, combine='nested', concat_dim='Time')
    
    ltc_daterange = pd.date_range(start=startdate, end=enddate, freq='h')
    # 去除时间序列的时区信息 - Remove timezone information
    time_index = pd.DatetimeIndex([t.tz_localize(None) for t in ltc_daterange])
    dataset = dataset.assign_coords(Time=('Time', time_index))
    logger.info("Time coordinates adjusted to local time")
    
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

def process_wrf_profile(datapath):
    """
    处理WRF剖面数据
    Process WRF profile data
    """
    # 合并WRF输出文件
    ds = combine_wrfout(datapath, START_DATE, END_DATE, DOMAIN)
    
    # 检测可用的化学变量
    available_chem_vars = detect_available_chemical_vars(ds) if CHEM else {}
    
    # 获取经纬度数据
    lon = ds['XLONG'][0, :, :].squeeze()
    lat = ds['XLAT'][0, :, :].squeeze()
    
    # 查找最近网格点
    nr_point = nearest_position(LONGITUDE, LATITUDE, lon.values, lat.values)
    
    # 提取基础变量
    logger.info("Processing meteorological variables")
    dataset = ds[met_vars].isel(south_north=nr_point[0], west_east=nr_point[1])
    
    # 计算衍生变量
    dataset['Temp'] = calculate_temperature(dataset)
    dataset['RH'] = calculate_relative_humidity(dataset)
    dataset['Pres'] = (dataset['P'] + dataset['PB']) / 100  # 转换为hPa
    dataset['Height'] = (dataset['PH'] + dataset['PHB']) / 9.8
    
    # 处理U/V变量（注意交错网格）
    lon_u = ds['XLONG_U'][0, :, :].squeeze()
    lat_u = ds['XLAT_U'][0, :, :].squeeze()
    lon_v = ds['XLONG_V'][0, :, :].squeeze()
    lat_v = ds['XLAT_V'][0, :, :].squeeze()
    nr_point_u = nearest_position(LONGITUDE, LATITUDE, lon_u.values, lat_u.values)
    nr_point_v = nearest_position(LONGITUDE, LATITUDE, lon_v.values, lat_v.values)
    dataset['U'] = ds['U'].isel(south_north=nr_point_u[0], west_east_stag=nr_point_u[1]).squeeze()
    dataset['V'] = ds['V'].isel(south_north_stag=nr_point_v[0], west_east=nr_point_v[1]).squeeze()
    dataset['WS'] = (dataset['U']**2 + dataset['V']**2)**0.5
    
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
    
    return dataset.drop_vars(rm_coords, errors='ignore')

def convert_to_excel(dataset, output_file):
    """
    将数据集转换为Excel文件
    Convert dataset to Excel file
    """
    logger.info(f"Converting to Excel: {output_file}")
    
    output_vars = ['Temp', 'RH', 'Pres', 'U', 'V', 'WS', 'Height']
    if CHEM:
        output_vars.extend(list(CHEMICAL_VARS.keys()))
    
    with pd.ExcelWriter(output_file) as writer:
        vars_missing = 0
        for var in output_vars:
            if var in dataset:
                # 转换为DataFrame并处理多级索引
                df = dataset[var].to_dataframe().unstack(level=1)
                df.columns = df.columns.droplevel(0)
                
                # 确保索引没有时区信息
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                df.to_excel(writer, sheet_name=var)
                logger.info(f"Exported {var}")
            else:
                vars_missing += 1
                logger.warning(f"Variable '{var}' not in dataset")
    
    if vars_missing > 0:
        logger.warning(f"{vars_missing} variables were not available")

def main():
    prog_start_time = time.time()
    
    logger.info(f"Processing profile data from {START_DATE} to {END_DATE}")
    logger.info(f"Location: ({LONGITUDE}, {LATITUDE}) in domain {DOMAIN}")
    
    # 处理WRF剖面数据
    dataset = process_wrf_profile(WRFOUT_DIR)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # 导出数据
    convert_to_excel(dataset, output_file)
    
    prog_end_time = time.time()
    elapsed_time = prog_end_time - prog_start_time
    if elapsed_time < 60:
        logger.info(f"Completed in {elapsed_time:.2f} seconds")
    elif elapsed_time < 3600:
        logger.info(f"Completed in {elapsed_time / 60:.2f} minutes")
    else:
        logger.info(f"Completed in {elapsed_time / 3600:.2f} hours")
    
if __name__ == "__main__":
    main() 