#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
'''
This script is used to get single-level spatial data from wrfout files
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
OUTPUT_FILENAME = 'sfc.nc'

# Time Range Settings
START_DATE = '2024-07-22 00:00:00'
END_DATE = '2024-09-08 00:00:00'

# Level Settings
LEVEL = 0  # 0: surface, 2: 100m, 11: 500m, 19: 1000m

# Whether to Process Chemical Species
CHEM = True

CHEMICAL_VARS = {
# 'var_name in output.nc': 'var_name in wrfout'
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
LOG_FILE = "get_spatial_wrfout.log"

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

ground_vars = ['T2', 'Q2', 'PSFC', 'U10', 'V10']
upper_vars = ['T', 'QVAPOR', 'P', 'PB', 'PH', 'PHB']

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
        timelist(wyoming模式): 时间列表 (Time list, for wyoming mode)
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
    
    dataset = xr.open_mfdataset(filelist,combine='nested', concat_dim='Time')
    
    ltc_daterange = pd.date_range(start=startdate, end=enddate, freq='h')
    # 去除时间序列的时区信息 - Remove timezone information
    time_index = pd.DatetimeIndex([t.tz_localize(None) for t in ltc_daterange])
    dataset = dataset.assign_coords(Time=('Time', time_index))
    logger.info("Time coordinates adjusted to local time")
    
    return dataset

def detect_available_chemical_vars(dataset):
    """
    检测数据集中可用的化学变量
    Detect available chemical variables in the dataset
    """
    available_vars = {}
    found_count = 0

    for output_var, wrfout_var in CHEMICAL_VARS.items():
        if wrfout_var == 'bc':
            bc_vars = [key for key in dataset.keys() if re.match(r'bc_a\d+', key)]
            logger.info(f"Found {len(bc_vars)} bc_a* variables -> {output_var}")
            found_count += len(bc_vars)
        elif wrfout_var in dataset:
            found_count += 1
            available_vars[output_var] = wrfout_var
            logger.info(f"Found variable: {wrfout_var} -> {output_var}")
        else:
            logger.warning(f"Cannot find variable: {wrfout_var}")
    return available_vars

def process_wrf_sfc(datapath):
    """
    处理WRF地面空间数据
    Process WRF surface spatial data
    """
    ds = combine_wrfout(datapath, START_DATE, END_DATE, DOMAIN)

    dataset = ds[ground_vars].squeeze()
    dataset['lon'] = ds['XLONG'][0, :, :].squeeze()
    dataset['lat'] = ds['XLAT'][0, :, :].squeeze()
    
    # Process meteorological variables
    logger.info("Processing meteorological variables")
    dataset['Temp'] = dataset['T2'] - 273.15
    dataset['RH'] = calculate_rh_2d(dataset)
    dataset['Pres'] = dataset['PSFC'] / 100
    dataset['WS'] = (dataset['U10'] ** 2 + dataset['V10'] ** 2) ** 0.5
    dataset['U'] = dataset['U10']
    dataset['V'] = dataset['V10']
    
    dataset = dataset.drop_vars(ground_vars, errors='ignore')
    
    if CHEM:
        # Process chemical variables
        logger.info("Processing chemical variables")
        available_vars = detect_available_chemical_vars(ds)
        for output_var, wrfout_var in available_vars.items():
            dataset[output_var] = ds[wrfout_var].sel(bottom_top=0).squeeze()
        if 'bc' in CHEMICAL_VARS.values():
            bc_vars = [key for key in ds.keys() if re.match(r'bc_a\d+', key)]
            dataset['BC'] = sum(ds[key].sel(bottom_top=0).squeeze() for key in bc_vars)
    
    return dataset.drop_vars(rm_coords, errors='ignore')
    
def process_wrf_upper(datapath, level=LEVEL):
    """
    处理WRF不同高度水平空间数据
    Process WRF horizontal spatial data at different heights
    """
    ds = combine_wrfout(datapath, START_DATE, END_DATE, DOMAIN)

    dataset = ds[upper_vars].squeeze()
    dataset['lon'] = ds['XLONG'][0, :, :].squeeze()
    dataset['lat'] = ds['XLAT'][0, :, :].squeeze()

    # Process meteorological variables
    logger.info("Processing meteorological variables")
    dataset['Temp'] = calculate_temperature(dataset)
    dataset['RH'] = calculate_relative_humidity(dataset)
    dataset['Pres'] = (dataset['P'] + dataset['PB']) / 100
    dataset['Height'] = (dataset['PH'] + dataset['PHB']) / 9.8
    
    dataset['Temp'] = dataset['Temp'].sel(bottom_top=level)
    dataset['RH'] = dataset['RH'].sel(bottom_top=level)
    dataset['Pres'] = dataset['Pres'].sel(bottom_top=level)
    dataset['Height'] = dataset['Height'].sel(bottom_top_stag=level)

    dataset = dataset.drop_vars(upper_vars, errors='ignore')
    
    # wind
    UU = 0.5 * (ds['U'].sel(west_east_stag=slice(0, -1)) + ds['U'].sel(west_east_stag=slice(1, None))).sel(bottom_top=level)
    VV = 0.5 * (ds['V'].sel(south_north_stag=slice(0, -1)) + ds['V'].sel(south_north_stag=slice(1, None))).sel(bottom_top=level)
    dataset['U'] = UU.rename({'west_east_stag': 'west_east'})
    dataset['V'] = VV.rename({'south_north_stag': 'south_north'})
    dataset['WS'] = (dataset['U'] ** 2 + dataset['V'] ** 2) ** 0.5
    
    if CHEM:
        # Process chemical variables
        logger.info("Processing chemical variables")
        available_vars = detect_available_chemical_vars(ds)
        for output_var, wrfout_var in available_vars.items():
            dataset[output_var] = ds[wrfout_var].sel(bottom_top=level).squeeze()
        if 'bc' in CHEMICAL_VARS.values():
            bc_vars = [key for key in ds.keys() if re.match(r'bc_a\d+', key)]
            dataset['BC'] = sum(ds[key].sel(bottom_top=level).squeeze() for key in bc_vars)
        
    return dataset.drop_vars(rm_coords, errors='ignore')

def convert_to_netcdf(dataset, output_file):
    """
    将数据集转换为netcdf文件
    Convert dataset to netcdf file
    """
    logger.info(f"Converting to NetCDF: {os.path.basename(output_file)}")
    
    if 'Time' in dataset.coords:
        if hasattr(dataset.Time, 'dt'):
            if any(hasattr(t, 'tz') and t.tz is not None for t in dataset.Time.values if hasattr(t, 'tz')):
                logger.info("Removing timezone information for NetCDF compatibility")
                dataset['Time'] = pd.DatetimeIndex([pd.Timestamp(t).tz_localize(None) if hasattr(t, 'tz') and t.tz is not None else t for t in dataset.Time.values])
    
    encoding = {
        var: {'zlib': True, 'complevel': 4}
        for var in dataset.data_vars
    }
    
    try:
        dataset.to_netcdf(output_file, encoding=encoding)
        logger.info(f"Data saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving NetCDF: {e}")
        raise e
    

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

def calculate_rh_2d(dataset):
    """
    计算地面相对湿度
    Calculate surface relative humidity
    """
    t2 = dataset['T2'] - 273.15
    q2 = dataset['Q2']
    p = dataset['PSFC'] / 100  # 压力, hPa (pressure, hPa)
    e = p * q2 / (0.622 + q2)
    es = 6.112 * np.exp(17.67 * t2 / (t2 + 243.5))
    rh = e / es * 100
    return rh

def main():
    
    prog_start_time = time.time()
    
    logger.info(f"Processing data from {START_DATE} to {END_DATE}")
    
    if LEVEL == 0:
        logger.info("Processing surface data")
        dataset = process_wrf_sfc(WRFOUT_DIR)
    else:
        logger.info(f"Processing data at level={LEVEL}")
        dataset = process_wrf_upper(WRFOUT_DIR, LEVEL)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    convert_to_netcdf(dataset, output_file)
    
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