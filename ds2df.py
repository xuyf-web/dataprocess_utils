import os
import re
import time
import numpy as np
import xarray as xr
import pandas as pd
import dask
from namelist import DATA_PATH
from findpoint import nearest_position


#########################
# 用户配置选项（可根据需要修改）
# User Configuration Options (Modify as needed)
#########################

# 处理模式选择
# Processing Mode Selection
MODE = 'wyoming'    # 'profile'（提取单站点垂直数据） (extract vertical data for single station),
                    # 'grd'（提取地面网格数据）(extract surface grid data),
                    # 'wyoming'（提取多站点数据）(extract multi-station data)

# 时间范围设置
# Time Range Settings
start_time = '2024-04-02 00:00:00'
end_time = '2024-04-14 00:00:00'

# 是否处理化学成分
# Whether to Process Chemical Species
process_chemistry = False

# 化学成分变量定义 (修改这里以调整处理的化学变量)
# Chemical Species Variables (Modify here to adjust processed chemical variables)
# 注意：这里的变量名应与WRF输出文件中的变量名一致
# Note: These variable names should match those in the WRF output files
chemical_vars = {
    'o3': 'O3',      # 臭氧 (Ozone)
    'no2': 'NO2',    # 二氧化氮 (Nitrogen dioxide)
    'no': 'NO',      # 一氧化氮 (Nitric oxide)
    'co': 'CO',      # 一氧化碳 (Carbon monoxide)
    # 异戊二烯在不同化学机制中有不同命名:
    # Isoprene has different names in different chemical mechanisms:
    # 'isoprene': SAPRC99机制 (SAPRC99 mechanism)
    # 'iso': CBMZ/CBMZ-MOSAIC机制 (CBMZ/CBMZ-MOSAIC mechanism)
    'isoprene': 'ISOP',  # 异戊二烯 (Isoprene) - SAPRC99机制
    'iso': 'ISOP',       # 异戊二烯 (Isoprene) - CBMZ/CBMZ-MOSAIC机制
    # 黑碳会自动从 bc_a0* 变量中计算 (Black carbon will be automatically calculated from bc_a0* variables)
}

# 目标化学成分变量 (修改这里以调整输出的化学变量)
# Target Chemical Species (Modify here to adjust output chemical variables)
# 这些是输出文件中的变量名称
# These are the variable names in the output files
target_chemical_vars = ['O3', 'NO2', 'NO', 'CO', 'ISOP', 'BC']

# 输出目录
# Output Directory
# output_dir = os.path.join(DATA_PATH, "mytest/postwrf/met")
# output_dir = os.path.join(DATA_PATH, "mytest/postwrf/chem")
output_dir = os.path.join(DATA_PATH, "mytest/postwrf/wyoming")

# 数据源配置（根据需求修改）
# Data Source Configuration (Modify as needed)
datasets = [
    # (wrfout_path, domain, output_filename)
    # (f"{DATA_PATH}mytest/met1", 'd02', 'met1'),
    # (f"{DATA_PATH}mytest/met2", 'd02', 'met2'),
    # (f"{DATA_PATH}Zhouxy", 'd01', 'cbmz'),
    # (f"{DATA_PATH}mytest/chem1", 'd02', 'chem1'),
    (f"{DATA_PATH}mytest/metnew1", 'd01', 'metnew1'),
    (f"{DATA_PATH}mytest/metnew2", 'd02', 'metnew2'),
    (f"{DATA_PATH}mytest/metnew3", 'd01', 'metnew3'),
    (f"{DATA_PATH}mytest/metnew4", 'd02', 'metnew4'),
]

# 单站点模式（profile模式）的站点设置
# Single Station Settings (for profile mode)
std_lon = 116.72
std_lat = 32.59

# 多站点模式（wyoming模式）的站点设置
# Multi-Station Settings (for wyoming mode)
station_location = {
    # 'STATION_NAME': [LON, LAT]
    'NANJING': [118.9, 31.93],
    'SHANGHAI': [121.45, 31.42],
    'HANGZHOU': [120.17, 30.23],
    'QUXIAN': [118.87, 28.97],
    'ANQING': [116.97, 30.62],
    'FUYANG': [115.73, 32.87],
    'XUZHOU': [117.15, 34.28],
    'SHEYANG': [120.3, 33.75],
}

# 配置dask以避免内存问题
dask.config.set({"array.chunk-size": "128MiB"})

# 为输出增加颜色
class Colors:
    BLUE = '\033[94m'      # 信息
    GREEN = '\033[92m'     # 成功
    YELLOW = '\033[93m'    # 警告
    RED = '\033[91m'       # 错误
    CYAN = '\033[96m'      # 变量信息
    MAGENTA = '\033[95m'   # 路径信息
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# 打印函数 - 添加详细级别控制
# 0: 只显示错误, 1: 显示错误和警告, 2: 显示所有重要信息, 3: 显示所有信息
VERBOSE_LEVEL = 2  # 修改为2，保留重要信息

def print_info(message, level=2):
    if VERBOSE_LEVEL >= level:
        print(f"{Colors.BLUE}INFO: {message}{Colors.END}")

def print_success(message, level=2):
    if VERBOSE_LEVEL >= level:
        print(f"{Colors.GREEN}SUCCESS: {message}{Colors.END}")

def print_warning(message, level=1):
    if VERBOSE_LEVEL >= level:
        print(f"{Colors.YELLOW}WARNING: {message}{Colors.END}")

def print_error(message, level=0):
    if VERBOSE_LEVEL >= level:
        print(f"{Colors.RED}ERROR: {message}{Colors.END}")

def print_var(message, level=3):
    if VERBOSE_LEVEL >= level:
        print(f"{Colors.CYAN}{message}{Colors.END}")

def print_path(message, level=3):
    if VERBOSE_LEVEL >= level:
        print(f"{Colors.MAGENTA}{message}{Colors.END}")

def print_section(message, level=2):
    if VERBOSE_LEVEL >= level:
        print(f"\n{Colors.BOLD}{Colors.UNDERLINE}{message}{Colors.END}")

#####################
# 变量定义
# Variable Definitions
#####################

# 处理变量定义
# Processing Variable Definitions
met_vars = ['T', 'QVAPOR', 'P', 'PB', 'PH', 'PHB']
wind_vars = ['U', 'V']
ground_vars = ['T2', 'Q2', 'PSFC', 'U10', 'V10']

# 根据处理模式和化学成分设置变量
# Set variables based on processing mode and chemical species options
if MODE == 'grd':
    if process_chemistry:
        use_vars = ground_vars + list(chemical_vars.keys())
    else:
        use_vars = ground_vars
    target_vars = ['lon', 'lat', 'Temp', 'RH', 'Pres', 'WS']
else:  # profile 或 wyoming 模式 (profile or wyoming mode)
    if process_chemistry:
        use_vars = met_vars + list(chemical_vars.keys())
    else:
        use_vars = met_vars
    target_vars = ['Temp', 'RH', 'Pres', 'WS', 'Height']

# 添加化学变量到目标变量
# Add chemical variables to target variables
if process_chemistry:
    target_vars.extend(target_chemical_vars)

# 需要移除的变量和坐标
# Variables and coordinates to remove
rm_vars = met_vars + wind_vars
if process_chemistry:
    rm_vars += list(chemical_vars.keys())
rm_coords = ['XLONG', 'XLAT', 'XLONG_U', 'XLAT_U', 'XLONG_V', 'XLAT_V', 'XTIME']

#####################
# 功能函数
# Function Definitions
#####################

def generate_filelist(datapath, start, end, domain):
    """
    生成WRF输出文件列表
    Generate WRF output file list
    
    Args:
        datapath: WRF输出文件路径 (WRF output file path)
        start: 开始时间 (Start time)
        end: 结束时间 (End time)
        domain: 模拟域 (Simulation domain)
    Returns:
        filelist: 文件列表 (File list)
        timelist(wyoming模式): 时间列表 (Time list, for wyoming mode)
    """
    if not datapath.endswith(os.sep):
        datapath += os.sep
        
    if MODE == 'wyoming':
        # Wyoming模式下，只处理00:00和12:00的数据
        # In Wyoming mode, only process 00:00 and 12:00 data
        startdate = pd.Timestamp(start).tz_localize('Asia/Shanghai')
        enddate = pd.Timestamp(end).tz_localize('Asia/Shanghai')
        date_range = pd.date_range(start=startdate, end=enddate, freq='D')
        
        # 创建00:00和12:00数据的文件路径列表
        # Create file path list for 00:00 and 12:00 data
        filelist = []
        timelist = []
        for day in date_range:
            # 本地时变为UTC后检查wrfout是否存在
            # Convert local time to UTC and check if wrfout exists
            local00_to_utc = day.tz_convert('UTC')
            # 使用符合文件名约定的时间格式
            utc_filename_format = local00_to_utc.strftime('%Y-%m-%d_%H:%M:%S')
            # 不要替换时间格式中的冒号
            filename00 = f"wrfout_{domain}_{utc_filename_format}"
            full_path00 = os.path.join(datapath, filename00)
            
            if os.path.exists(full_path00):
                filelist.append(full_path00)
                timelist.append(local00_to_utc)
                # 文件找到的信息设为高级别，不频繁显示
                print_info(f"Found file: {os.path.basename(full_path00)}", 3)
            else:
                print_warning(f"File not found: {os.path.basename(full_path00)}")
            
            local12_to_utc = (day + pd.Timedelta(hours=12)).tz_convert('UTC')
            # 使用符合文件名约定的时间格式
            utc_filename_format = local12_to_utc.strftime('%Y-%m-%d_%H:%M:%S')
            # 不要替换时间格式中的冒号
            filename12 = f"wrfout_{domain}_{utc_filename_format}"
            full_path12 = os.path.join(datapath, filename12)
            
            if os.path.exists(full_path12):
                filelist.append(full_path12)
                timelist.append(local12_to_utc)
                # 文件找到的信息设为高级别，不频繁显示
                print_info(f"Found file: {os.path.basename(full_path12)}", 3)
            else:
                print_warning(f"File not found: {os.path.basename(full_path12)}")
        
        if not filelist:
            print_error(f"No valid files found in {datapath} for domain {domain}")
            print_info(f"Looking for files matching pattern: wrfout_{domain}_YYYY-MM-DD_HH:MM:SS")
        else:
            print_success(f"Found {len(filelist)} files for processing")
        
        return filelist, timelist
    else:
        # 其他模式下，使用逐小时数据
        # In other modes, use hourly data
        startdate = pd.Timestamp(start).tz_localize('Asia/Shanghai')
        enddate = pd.Timestamp(end).tz_localize('Asia/Shanghai')
        
        # 所有模式都进行时区转换
        # All modes need timezone conversion
        startutc = startdate.tz_convert('UTC')
        endutc = enddate.tz_convert('UTC')
        date_range = pd.date_range(start=startutc, end=endutc, freq='h')
        
        # 创建符合文件名约定的文件路径列表
        filelist = []
        file_count = 0
        missing_count = 0
        for d in date_range:
            utc_filename_format = d.strftime('%Y-%m-%d_%H:%M:%S')
            # 不要替换时间格式中的冒号
            filename = f"wrfout_{domain}_{utc_filename_format}"
            full_path = os.path.join(datapath, filename)
            if os.path.exists(full_path):
                filelist.append(full_path)
                file_count += 1
                # 不输出每个文件的信息，过于冗长
            else:
                missing_count += 1
                # 不输出每个未找到的文件，过于冗长
        
        if not filelist:
            print_error(f"No valid files found in {datapath} for domain {domain}")
            print_info(f"Looking for files matching pattern: wrfout_{domain}_YYYY-MM-DD_HH:MM:SS")
        else:
            print_success(f"Found {file_count} files for processing")
            if missing_count > 0:
                print_warning(f"Missing {missing_count} expected files")
            
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
    print_info(f"Reading data from {os.path.basename(datapath)}")
    
    if MODE == 'wyoming':
        filelist, timelist = generate_filelist(datapath, start, end, domain)
        
        dataset = xr.open_mfdataset(filelist,
                                   combine='nested', concat_dim='Time',
                                   engine='netcdf4',
                                   parallel=True)
        
        # 调整UTC到本地时间
        # Convert UTC to local time
        local_times = [t.tz_convert('Asia/Shanghai').tz_localize(None) for t in timelist]
        dataset = dataset.assign_coords(Time=('Time', local_times))
        print_info("Timezone converted: UTC to Asia/Shanghai")
    else:
        filelist = generate_filelist(datapath, start, end, domain)
        
        dataset = xr.open_mfdataset(filelist, combine='nested', concat_dim='Time')
        
        # 时间调整 - 所有模式使用相同的时间处理
        # Time adjustment - all modes use the same time processing
        startdate = pd.Timestamp(start).tz_localize('Asia/Shanghai')
        enddate = pd.Timestamp(end).tz_localize('Asia/Shanghai')
        ltc_daterange = pd.date_range(start=startdate, end=enddate, freq='h')
        # 确保时间序列没有时区信息 - Convert to timezone-naive datetime for NetCDF compatibility
        time_index = pd.DatetimeIndex([t.tz_localize(None) for t in ltc_daterange])
        print_info("Time coordinates adjusted to local time")
        dataset = dataset.assign_coords(Time=('Time', time_index))
    
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

def detect_available_chemical_vars(ds):
    """
    检测数据集中可用的化学变量
    Detect available chemical variables in the dataset
    
    Args:
        ds: 数据集 (Dataset)
    Returns:
        available_vars: 可用变量字典 (Available variables dictionary)
    """
    available_vars = {}
    found_count = 0
    
    # 检查每个化学变量 (Check each chemical variable)
    for var, target_var in chemical_vars.items():
        if var in ds:
            # 如果变量存在于数据集中 (If variable exists in dataset)
            found_count += 1
            if target_var not in available_vars:
                available_vars[var] = target_var
                # 对于每个化学组分名称只输出一次
                print_var(f"Found chemical variable: {var} -> {target_var}")
            elif target_var == 'ISOP':
                # 如果是异戊二烯，记录下已找到的变量名 (If isoprene, record the found variable name)
                available_vars[var] = target_var
                print_var(f"Found isoprene variant: {var}")
    
    return available_vars

def process_wrf_data_grd(simpath, domain='d02'):
    """
    处理WRF地面数据（grd模式）
    Process WRF surface data (grd mode)
    """
    # 合并WRF输出文件 (Combine WRF output files)
    print_section(f"Processing ground data from {os.path.basename(simpath)}")
    ds = combine_wrfout(simpath, start_time, end_time, domain)
    
    # 检测可用的化学变量 (Detect available chemical variables)
    available_chem_vars = detect_available_chemical_vars(ds) if process_chemistry else {}
    if available_chem_vars:
        print_info(f"Processing {len(available_chem_vars)} chemical variables")
    else:
        print_warning("No chemical variables found")

    # 根据可用变量调整use_vars
    # Adjust use_vars based on available variables
    actual_use_vars = ground_vars.copy()
    if process_chemistry:
        actual_use_vars.extend(available_chem_vars.keys())
    print_info(f"Extracting required variables")

    # 提取基础变量 (Extract basic variables)
    dataset = ds[actual_use_vars].squeeze()
    dataset['lon'] = ds['XLONG'][0, :, :].squeeze()
    dataset['lat'] = ds['XLAT'][0, :, :].squeeze()

    # 处理目标变量 (Process target variables)
    print_info("Processing meteorological variables")
    dataset['Temp'] = dataset['T2'] - 273.15
    dataset['RH'] = calculate_rh_2d(dataset)
    dataset['Pres'] = dataset['PSFC'] / 100
    dataset['WS'] = (dataset['U10'] ** 2 + dataset['V10'] ** 2) ** 0.5

    if process_chemistry:
        # 处理化学变量 (Process chemical variables)
        print_info("Processing chemical variables")
        chemical_vars_processed = 0
        for var, target_var in available_chem_vars.items():
            dataset[target_var] = dataset[var].sel(bottom_top=0).squeeze() * 1000
            chemical_vars_processed += 1
        print_success(f"Processed {chemical_vars_processed} chemical variables")

        # 黑碳处理 (Black carbon processing)
        bc_vars = [key for key in ds.keys() if re.match(r'bc_a0\d+', key)]
        if bc_vars:
            dataset['BC'] = sum(ds[key].sel(bottom_top=0).squeeze() for key in bc_vars)
            print_success(f"Processed black carbon from {len(bc_vars)} variables")
        else:
            print_warning("Cannot find BC variables")
            if any(v in dataset for v in ['O3', 'NO2', 'NO', 'CO', 'ISOP']):
                # Use any available chemical variable as template
                for v in ['O3', 'NO2', 'NO', 'CO', 'ISOP']:
                    if v in dataset:
                        dataset['BC'] = np.zeros_like(dataset[v])
                        print_info(f"Created zero-filled BC")
                        break

    # 确保时间坐标没有时区信息 (Ensure time coordinate has no timezone info)
    if 'Time' in dataset.coords:
        if hasattr(dataset.Time, 'dt'):
            # 检查是否有时区信息
            if hasattr(dataset.Time.values[0], 'tz') and dataset.Time.values[0].tz is not None:
                # 移除时区信息
                print_info("Removing timezone information from time coordinates")
                dataset['Time'] = dataset.Time.dt.tz_localize(None)
    
    # 移除原始变量 (Remove original variables)
    print_info("Cleaning dataset by removing original variables")
    dataset = dataset.drop_vars(ground_vars, errors='ignore')
    if process_chemistry:
        dataset = dataset.drop_vars(list(available_chem_vars.keys()), errors='ignore')
    return dataset.drop_vars(rm_coords, errors='ignore')

def process_wrf_data_profile(simpath, domain='d02'):
    """
    处理WRF剖面数据（profile模式）
    Process WRF profile data (profile mode)
    """
    # 合并WRF输出文件 (Combine WRF output files)
    ds = combine_wrfout(simpath, start_time, end_time, domain)
    
    # 检测可用的化学变量 (Detect available chemical variables)
    available_chem_vars = detect_available_chemical_vars(ds) if process_chemistry else {}
    print(f"Available chemical variables: {', '.join(available_chem_vars.keys())}")
    
    # 根据可用变量调整use_vars
    # Adjust use_vars based on available variables
    actual_use_vars = met_vars.copy()
    if process_chemistry:
        actual_use_vars.extend(available_chem_vars.keys())
    
    # 获取经纬度数据 (Get longitude and latitude data)
    lon = ds['XLONG'][0, :, :].squeeze()
    lat = ds['XLAT'][0, :, :].squeeze()
    
    # 查找最近网格点 (Find nearest grid point)
    nr_point = nearest_position(std_lon, std_lat, lon.values, lat.values)
    
    # 提取基础变量 (Extract basic variables)
    dataset_base = ds[actual_use_vars].isel(south_north=nr_point[0], west_east=nr_point[1])
    
    # 处理U/V变量（注意交错网格）(Process U/V variables, note staggered grid)
    dataset_u = ds['U'].isel(south_north=nr_point[0], west_east_stag=nr_point[1])
    dataset_v = ds['V'].isel(south_north_stag=nr_point[0], west_east=nr_point[1])
    
    # 合并数据集 (Merge datasets)
    dataset = xr.merge([dataset_base, dataset_u, dataset_v])
    
    # 计算衍生变量 (Calculate derived variables)
    dataset['Temp'] = calculate_temperature(dataset)
    dataset['RH'] = calculate_relative_humidity(dataset)
    dataset['Pres'] = (dataset['P'] + dataset['PB']) / 100  # 转换为hPa (Convert to hPa)
    dataset['WS'] = (dataset['U']**2 + dataset['V']**2)**0.5
    dataset['Height'] = (dataset['PH'] + dataset['PHB']) / 9.8
    
    if process_chemistry:
        # 处理化学变量 (Process chemical variables)
        for var, target_var in available_chem_vars.items():
            dataset[target_var] = dataset[var] * 1000
            print(f"Processed chemical variable: {var} -> {target_var}")
        
        # 黑碳处理 (Black carbon processing)
        bc_vars = [key for key in ds.keys() if re.match(r'bc_a0\d+', key)]
        if bc_vars:
            dataset['BC'] = sum(ds[key].isel(south_north=nr_point[0], west_east=nr_point[1]) for key in bc_vars)
            print(f"Processed black carbon from {len(bc_vars)} variables")
        else:
            print("Warning: Cannot find black carbon variables (bc_a0*)")
            if any(v in dataset for v in ['O3', 'NO2', 'NO', 'CO', 'ISOP']):
                # Use any available chemical variable as template
                for v in ['O3', 'NO2', 'NO', 'CO', 'ISOP']:
                    if v in dataset:
                        dataset['BC'] = np.zeros_like(dataset[v])
                        print(f"Created zero-filled BC using {v} as template")
                        break
    
    # 清理原始变量 (Clean up original variables)
    dataset = dataset.drop_vars(met_vars + wind_vars, errors='ignore')
    if process_chemistry:
        dataset = dataset.drop_vars(list(available_chem_vars.keys()), errors='ignore')
    return dataset.drop_vars(rm_coords, errors='ignore')

def process_wrf_data_wyoming(simpath, station_location, domain='d02'):
    """
    处理WRF多站点数据（wyoming模式）
    Process WRF multi-station data (wyoming mode)
    """
    # 合并WRF输出文件 (Combine WRF output files)
    ds = combine_wrfout(simpath, start_time, end_time, domain)
    
    # 检测可用的化学变量 (Detect available chemical variables)
    available_chem_vars = detect_available_chemical_vars(ds) if process_chemistry else {}
    print(f"Available chemical variables: {', '.join(available_chem_vars.keys())}")
    
    # 根据可用变量调整use_vars
    # Adjust use_vars based on available variables
    actual_use_vars = met_vars.copy()
    if process_chemistry:
        actual_use_vars.extend(available_chem_vars.keys())
    
    # 获取经纬度网格数据 (Get longitude and latitude grid data)
    lon = ds['XLONG'][0, :, :].squeeze()
    lat = ds['XLAT'][0, :, :].squeeze()
    
    # 存储各站点数据集 (Store datasets for each station)
    station_datasets = {}
    
    # 循环处理每个站点 (Process each station)
    for station_name, (std_lon, std_lat) in station_location.items():
        print(f"Processing station: {station_name}")
        # 查找最近网格点 (Find nearest grid point)
        nr_point = nearest_position(std_lon, std_lat, lon.values, lat.values)
        
        # 提取基础变量 (Extract basic variables)
        dataset_base = ds[actual_use_vars].isel(south_north=nr_point[0], west_east=nr_point[1])
        
        # 处理交错网格变量 (Process staggered grid variables)
        dataset_u = ds['U'].isel(south_north=nr_point[0], west_east_stag=nr_point[1])
        dataset_v = ds['V'].isel(south_north_stag=nr_point[0], west_east=nr_point[1])
        
        # 合并基础数据集 (Merge basic datasets)
        dataset = xr.merge([dataset_base, dataset_u, dataset_v])
        
        # 计算衍生变量 (Calculate derived variables)
        dataset['Temp'] = calculate_temperature(dataset)
        dataset['RH'] = calculate_relative_humidity(dataset)
        dataset['Pres'] = (dataset['P'] + dataset['PB']) / 100  # 转换为hPa (Convert to hPa)
        dataset['WS'] = np.sqrt(dataset['U']**2 + dataset['V']**2)
        dataset['WD'] = np.degrees(np.arctan2(-dataset['U'], -dataset['V'])) % 360
        dataset['Height'] = (dataset['PH'] + dataset['PHB']) / 9.8
        
        # 处理化学变量 (Process chemical variables)
        if process_chemistry:
            for var, target_var in available_chem_vars.items():
                dataset[target_var] = dataset[var] * 1000
                print(f"Processed chemical variable for station {station_name}: {var} -> {target_var}")
                
            # 黑碳处理 (Black carbon processing)
            bc_vars = [v for v in ds.data_vars if re.match(r'bc_a0\d+', v)]
            if bc_vars:
                dataset['BC'] = sum(
                    ds[var].isel(south_north=nr_point[0], west_east=nr_point[1]) 
                    for var in bc_vars
                )
                print(f"Processed black carbon from {len(bc_vars)} variables for station {station_name}")
            else:
                print(f"Warning: Cannot find black carbon variables (bc_a0*) for station {station_name}")
                if any(v in dataset for v in ['O3', 'NO2', 'NO', 'CO', 'ISOP']):
                    # Use any available chemical variable as template
                    for v in ['O3', 'NO2', 'NO', 'CO', 'ISOP']:
                        if v in dataset:
                            dataset['BC'] = np.zeros_like(dataset[v])
                            print(f"Created zero-filled BC using {v} as template for station {station_name}")
                            break
        
        # 清理不需要的变量 (Clean up unnecessary variables)
        cleaned_dataset = dataset.drop_vars(met_vars + wind_vars, errors='ignore')
        if process_chemistry:
            cleaned_dataset = cleaned_dataset.drop_vars(list(available_chem_vars.keys()), errors='ignore')
        station_datasets[station_name] = cleaned_dataset.drop_vars(rm_coords, errors='ignore')
    
    print(f"Processing of wrfout data completed.")
    return station_datasets

def convert_to_excel(ds, output_file):
    """
    将数据集转换为Excel文件
    Convert dataset to Excel file
    """
    print_info(f"Converting to Excel: {os.path.basename(output_file)}")
    
    with pd.ExcelWriter(output_file) as writer:
        vars_exported = 0
        vars_missing = 0
        for var in target_vars:
            if var in ds:
                # 转换为DataFrame并处理多级索引 (Convert to DataFrame and handle multi-level index)
                df = ds[var].to_dataframe().unstack(level=1)
                df.columns = df.columns.droplevel(0)
                
                # 确保索引没有时区信息 (Ensure index has no timezone info)
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                df.to_excel(writer, sheet_name=var)
                vars_exported += 1
            else:
                vars_missing += 1
                print_warning(f"Variable '{var}' not in dataset")
    
    print_success(f"Exported {vars_exported} variables to {os.path.basename(output_file)}")
    if vars_missing > 0:
        print_warning(f"{vars_missing} variables were not available")

def convert_to_nc(ds, output_file):
    """
    将数据集转换为NetCDF文件
    Convert dataset to NetCDF file
    """
    print_info(f"Converting to NetCDF: {os.path.basename(output_file)}")
    
    # 创建一个副本，避免修改原始数据集
    ds_for_nc = ds.copy()
    
    # 确保数据集中的时间没有时区信息
    if 'Time' in ds_for_nc.coords:
        if hasattr(ds_for_nc.Time, 'dt'):
            # 检查是否有时区信息
            if any(hasattr(t, 'tz') and t.tz is not None for t in ds_for_nc.Time.values if hasattr(t, 'tz')):
                # 移除时区信息
                print_info("Removing timezone information for NetCDF compatibility")
                ds_for_nc['Time'] = pd.DatetimeIndex([pd.Timestamp(t).tz_localize(None) if hasattr(t, 'tz') and t.tz is not None else t for t in ds_for_nc.Time.values])
    
    # 设置压缩编码
    encoding = {
        var: {'zlib': True, 'complevel': 4}
        for var in ds_for_nc.data_vars
    }
    
    # 写入到NetCDF文件
    try:
        ds_for_nc.to_netcdf(output_file, encoding=encoding)
        print_success(f"Exported to {os.path.basename(output_file)}")
    except Exception as e:
        print_error(f"Error saving NetCDF: {str(e)}")
        # 尝试使用更安全的设置
        print_info("Trying with more compatible settings...")
        
        # 转换为更简单的数据结构
        if 'Time' in ds_for_nc.coords:
            ds_for_nc['Time'] = np.array([np.datetime64(t) for t in ds_for_nc.Time.values])
        
        # 不使用压缩
        ds_for_nc.to_netcdf(output_file)
        print_success(f"Exported to {os.path.basename(output_file)} (without compression)")

def main():
    """
    主函数
    Main function
    """
    prog_start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    print_section("WRF Data Processing")
    print_info(f"Mode: {MODE} | Chemistry: {'Yes' if process_chemistry else 'No'}")
    print_info(f"Time range: {start_time} to {end_time}")
    print_info(f"Output directory: {output_dir}")
    
    # 根据不同模式处理数据 (Process data according to different modes)
    dataset_count = 0
    for simpath, domain, filename in datasets:
        print_section(f"Processing dataset: {filename} (domain: {domain})")
        
        try:
            if MODE == 'grd':
                # 地面格点数据处理（导出为NC文件）(Process surface grid data, export to NC file)
                ds = process_wrf_data_grd(simpath, domain)
                output_file = os.path.join(output_dir, f"{filename}.nc")
                convert_to_nc(ds, output_file)
            
            elif MODE == 'profile':
                # 单站点垂直剖面处理（导出为Excel）(Process single station vertical profile, export to Excel)
                print_info(f"Processing profile data for location ({std_lon}, {std_lat})")
                ds = process_wrf_data_profile(simpath, domain)
                output_file = os.path.join(output_dir, f"{filename}.xlsx")
                convert_to_excel(ds, output_file)
            
            elif MODE == 'wyoming':
                # 多站点数据处理（导出为Excel）(Process multi-station data, export to Excel)
                station_count = len(station_location)
                print_info(f"Processing {station_count} stations")
                station_datasets = process_wrf_data_wyoming(simpath, station_location, domain)
                station_processed = 0
                for station, ds in station_datasets.items():
                    print_info(f"Processing station: {station}")
                    output_file = os.path.join(output_dir, f"{filename}_{station}.xlsx")
                    convert_to_excel(ds, output_file)
                    station_processed += 1
                print_success(f"Processed {station_processed} stations")
            else:
                print_error(f"Unknown mode: '{MODE}'")
                return
            
            dataset_count += 1
            print_success(f"Completed processing {os.path.basename(simpath)}")
        
        except Exception as e:
            print_error(f"Error processing {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 计算运行时间 (Calculate runtime)
    prog_end_time = time.time()
    elapsed_time = prog_end_time - prog_start_time

    print_section("Summary")
    print_success(f"Processed {dataset_count} datasets")
    if elapsed_time < 60:
        print_success(f"Completed in {elapsed_time:.2f} seconds")
    elif elapsed_time < 3600:
        print_success(f"Completed in {elapsed_time / 60:.2f} minutes")
    else:
        print_success(f"Completed in {elapsed_time / 3600:.2f} hours")

if __name__ == "__main__":
    main() 