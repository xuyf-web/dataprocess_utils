import os
import glob
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy import interpolate
from openpyxl.styles import PatternFill, Font

import warnings
warnings.filterwarnings("ignore")

# 参数设置
# Parameters settings
start_time = '2024-04-02 00:00:00'
end_time = '2024-04-14 00:00:00'

# 站点列表
# Station list
stations = ['SHEYANG', 'XUZHOU', 'FUYANG', 'ANQING', 'QUXIAN', 'HANGZHOU', 'SHANGHAI', 'NANJING']
station_codes = {
    'SHEYANG': '58150',
    'XUZHOU': '58027',
    'FUYANG': '58203',
    'ANQING': '58424',
    'QUXIAN': '58633',
    'HANGZHOU': '58457',
    'SHANGHAI': '58362',
    'NANJING': '58238'
}

# 模拟组
# Simulation cases
cases = ['metnew1', 'metnew2', 'metnew3', 'metnew4', 'cbmz']

# 要统计的变量
# Variables to analyze
variables = ['TEMP', 'RELH', 'PRES', 'SPED']  # 气温、相对湿度、气压、风速
sim_variables = {'TEMP': 'Temp', 'RELH': 'RH', 'PRES': 'Pres', 'SPED': 'WS'}  # 变量名映射

# 统计参数列表
# Statistics metrics list
stats_metrics = ['correlation', 'bias', 'rmse', 'mae', 'mean_obs', 'mean_sim', 'n_samples']

# 数据路径设置
# Data path settings
OBS_DIR = "/data8/xuyf/data/WyomingSounding/output"
SIM_DIR = "/data8/xuyf/Project/Shouxian/data/mytest/postwrf/wyoming"
OUTPUT_DIR = "/data8/xuyf/Project/Shouxian/figures/validation/stats"

# 确保输出目录存在
# Make sure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_obs_files(station_code, start_time_dt, end_time_dt):
    """
    获取指定站点和时间范围内的观测文件列表
    Get observation files for a specific station within the given time range
    
    Parameters:
    -----------
    station_code : str
        站点代码，如 '58150'
    start_time_dt : datetime
        开始时间
    end_time_dt : datetime
        结束时间
        
    Returns:
    --------
    files : list
        满足条件的文件路径列表
    """
    station_dir = f"{OBS_DIR}/{station_code}_{stations[list(station_codes.values()).index(station_code)]}"
    all_files = glob.glob(f"{station_dir}/*.csv")
    
    valid_files = []
    for file in all_files:
        # 从文件名解析日期时间
        # Parse datetime from filename
        filename = os.path.basename(file)
        date_str, hour_str = filename.replace('.csv', '').split('_')
        file_dt = dt.datetime.strptime(f"{date_str} {hour_str}", "%Y%m%d %H")
        
        if start_time_dt <= file_dt <= end_time_dt:
            valid_files.append(file)
    
    return valid_files

def load_obs_data(station, start_time_dt, end_time_dt):
    """
    加载指定站点的观测数据
    Load observation data for a specific station
    
    Parameters:
    -----------
    station : str
        站点名称，如 'SHEYANG'
    start_time_dt : datetime
        开始时间
    end_time_dt : datetime
        结束时间
        
    Returns:
    --------
    obs_data : dict
        以时间为键的观测数据字典
    """
    station_code = station_codes[station]
    files = get_obs_files(station_code, start_time_dt, end_time_dt)
    
    obs_data = {}
    for file in files:
        # 从文件名解析日期时间
        # Parse datetime from filename
        filename = os.path.basename(file)
        date_str, hour_str = filename.replace('.csv', '').split('_')
        file_dt = dt.datetime.strptime(f"{date_str} {hour_str}", "%Y%m%d %H")
        
        # 读取CSV文件
        # Read CSV file
        try:
            df = pd.read_csv(file)
            # 确保必要的列存在
            # Make sure necessary columns exist
            required_cols = ['PRES(hPa)', 'HGHT(m)', 'TEMP(C)', 'RELH(%)', 'SPED(m/s)']
            if all(col in df.columns for col in required_cols):
                # 重命名列以便于处理
                # Rename columns for easier processing
                df = df.rename(columns={
                    'PRES(hPa)': 'PRES',
                    'HGHT(m)': 'HGHT',
                    'TEMP(C)': 'TEMP',
                    'RELH(%)': 'RELH',
                    'SPED(m/s)': 'SPED'
                })
                
                # 删除包含缺失值的行
                # Remove rows with missing values
                df = df.dropna(subset=['PRES', 'HGHT', 'TEMP', 'RELH', 'SPED'])
                
                # 存储到字典中
                # Store in dictionary
                time_key = file_dt.strftime("%Y-%m-%d_%H:00")
                obs_data[time_key] = df
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
    
    return obs_data

def load_sim_data(station, case):
    """
    加载指定站点和模拟方案的模拟数据
    Load simulation data for a specific station and case
    
    Parameters:
    -----------
    station : str
        站点名称，如 'SHEYANG'
    case : str
        模拟方案，如 'metnew1'
        
    Returns:
    --------
    sim_data : dict
        模拟数据字典，按变量组织
    """
    sim_file = f"{SIM_DIR}/{case}_{station}.xlsx"
    
    sim_data = {}
    try:
        # 读取各个变量的数据
        # Read data for each variable
        for var in sim_variables.values():
            # 使用pandas读取excel中的指定sheet
            # Read specific sheet from excel file
            sheet_data = pd.read_excel(sim_file, sheet_name=var, index_col=0)
            sim_data[var] = sheet_data
        
        # 读取高度数据
        # Read height data
        height = pd.read_excel(sim_file, sheet_name='Height', index_col=0)
        data_z40 = height.values
        # 计算层中值
        # Calculate mid-level values
        data_z39 = 0.5 * (data_z40[:, 1:] + data_z40[:, :-1])
        sim_data['Height'] = pd.DataFrame(data_z39, index=height.index, columns=height.columns[:-1])
    
    except Exception as e:
        print(f"读取文件 {sim_file} 时出错: {e}")
        return None
    
    return sim_data

def interpolate_and_calculate_stats(station, obs_data, sim_data_dict):
    """
    将模拟数据按高度插值到观测高度，并计算统计指标
    Interpolate simulation data to observation heights and calculate statistics
    
    Parameters:
    -----------
    station : str
        站点名称
    obs_data : dict
        观测数据字典
    sim_data_dict : dict
        模拟数据字典，以case为键
        
    Returns:
    --------
    stats_results : dict
        统计结果字典
    """
    # 存储各变量的统计结果
    # Store statistics results for each variable
    var_stats = {var: [] for var in variables}
    
    # 对每个观测时间
    # For each observation time
    for time_key, obs_df in obs_data.items():
        # 转换时间格式 YYYY-MM-DD_HH:00 -> YYYY-MM-DD HH:00:00
        # Convert time format
        sim_time_key = time_key.replace('_', ' ') + ':00'
        
        # 对每个模拟方案
        # For each simulation case
        for case in cases:
            if case not in sim_data_dict or sim_data_dict[case] is None:
                continue
                
            sim_data = sim_data_dict[case]
            
            # 检查此时间点是否在模拟数据中
            # Check if this time is in simulation data
            if sim_time_key not in sim_data['Height'].index:
                continue
            
            # 对每个变量
            # For each variable
            for obs_var, sim_var in sim_variables.items():
                # 获取观测数据
                # Get observation data
                obs_heights = obs_df['HGHT'].values
                obs_values = obs_df[obs_var].values
                
                # 获取模拟数据
                # Get simulation data
                sim_heights = sim_data['Height'].loc[sim_time_key].values
                sim_values = sim_data[sim_var].loc[sim_time_key].values
                
                # 创建插值函数并插值
                # Create interpolation function and interpolate
                try:
                    f = interpolate.interp1d(
                        sim_heights, 
                        sim_values, 
                        kind='linear', 
                        bounds_error=False, 
                        fill_value='extrapolate'
                    )
                    
                    # 在观测高度上进行插值
                    # Interpolate at observation heights
                    interpolated_values = f(obs_heights)
                    
                    # 过滤掉NaN值
                    # Filter out NaN values
                    valid_indices = ~np.isnan(interpolated_values) & ~np.isnan(obs_values)
                    if np.sum(valid_indices) < 5:  # 至少需要5个有效点
                        continue
                        
                    valid_obs = obs_values[valid_indices]
                    valid_sim = interpolated_values[valid_indices]
                    
                    # 计算统计指标
                    # Calculate statistics
                    correlation = np.corrcoef(valid_obs, valid_sim)[0, 1] if len(valid_obs) > 1 else np.nan
                    bias = np.mean(valid_sim - valid_obs)
                    rmse = np.sqrt(np.mean((valid_sim - valid_obs) ** 2))
                    mae = np.mean(np.abs(valid_sim - valid_obs))
                    mean_obs = np.mean(valid_obs)
                    mean_sim = np.mean(valid_sim)
                    n_samples = len(valid_obs)
                    
                    # 存储结果
                    # Store results
                    var_stats[obs_var].append({
                        'station': station,
                        'case': case,
                        'time': time_key,
                        'correlation': correlation,
                        'bias': bias,
                        'rmse': rmse,
                        'mae': mae,
                        'mean_obs': mean_obs,
                        'mean_sim': mean_sim,
                        'n_samples': n_samples
                    })
                    
                except Exception as e:
                    print(f"插值错误 ({station}, {case}, {obs_var}, {time_key}): {e}")
    
    # 将结果转换为DataFrame
    # Convert results to DataFrame
    stats_df = {}
    for var in variables:
        if var_stats[var]:
            stats_df[var] = pd.DataFrame(var_stats[var])
    
    return stats_df

def calculate_average_stats(stats_df_dict):
    """
    计算平均统计指标
    Calculate average statistics
    
    Parameters:
    -----------
    stats_df_dict : dict
        每个变量的统计结果DataFrame字典
        
    Returns:
    --------
    avg_stats : dict
        平均统计结果字典
    """
    avg_stats = {}
    
    for var, stats_df in stats_df_dict.items():
        # 根据站点和模拟方案分组计算平均值
        # Group by station and case to calculate mean
        station_stats = []
        for station in stations:
            for case in cases:
                station_case_df = stats_df[(stats_df['station'] == station) & (stats_df['case'] == case)]
                
                if not station_case_df.empty:
                    avg_row = {
                        'station': station,
                        'case': case
                    }
                    
                    # 计算各统计指标的均值
                    # Calculate mean for each statistic metric
                    for metric in stats_metrics:
                        if metric == 'n_samples':
                            avg_row[metric] = station_case_df[metric].sum()
                        else:
                            avg_row[metric] = station_case_df[metric].mean()
                    
                    station_stats.append(avg_row)
        
        # 将站点平均结果转换为DataFrame
        # Convert station average results to DataFrame
        if station_stats:
            avg_stats[var] = pd.DataFrame(station_stats)
            
            # 计算每个模拟方案在所有站点上的平均值
            # Calculate average for each simulation case across all stations
            case_avg = []
            for case in cases:
                case_df = avg_stats[var][avg_stats[var]['case'] == case]
                
                if not case_df.empty:
                    case_row = {'case': case, 'station': 'ALL'}
                    for metric in stats_metrics:
                        if metric == 'n_samples':
                            case_row[metric] = case_df[metric].sum()
                        else:
                            case_row[metric] = case_df[metric].mean()
                    
                    case_avg.append(case_row)
            
            # 添加所有模拟方案的平均结果
            # Add average results for all simulation cases
            if case_avg:
                avg_stats[var] = pd.concat([avg_stats[var], pd.DataFrame(case_avg)])
    
    return avg_stats

def save_results_to_excel(stats_df_dict, avg_stats_dict):
    """
    将结果保存到Excel文件
    Save results to Excel file
    
    Parameters:
    -----------
    stats_df_dict : dict
        每个变量的详细统计结果
    avg_stats_dict : dict
        每个变量的平均统计结果
    """
    # 创建ExcelWriter对象
    # Create ExcelWriter object
    excel_path = os.path.join(OUTPUT_DIR, 'wyoming_sounding_stats.xlsx')
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    
    # 保存每个变量的平均统计结果
    # Save average statistics for each variable
    for var, avg_df in avg_stats_dict.items():
        sheet_name = f"{var}_avg"
        # 重排列顺序
        # Reorder columns
        cols = ['station', 'case'] + stats_metrics
        avg_df = avg_df[cols]
        avg_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # 保存每个变量的详细统计结果
    # Save detailed statistics for each variable
    for var, stats_df in stats_df_dict.items():
        sheet_name = f"{var}_details"
        # 重排列顺序
        # Reorder columns
        cols = ['station', 'case', 'time'] + stats_metrics
        stats_df = stats_df[cols]
        stats_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # 保存并关闭Excel文件
    # Save and close Excel file
    writer.close()
    print(f"结果已保存到: {excel_path}")

def main():
    """
    主函数
    Main function
    """
    # 解析时间范围
    # Parse time range
    start_time_dt = dt.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time_dt = dt.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    
    print(f"处理时间范围: {start_time} 至 {end_time}")
    print(f"处理站点: {', '.join(stations)}")
    print(f"处理模拟方案: {', '.join(cases)}")
    
    # 存储所有变量的统计结果
    # Store statistics for all variables
    all_stats = {}
    
    # 对每个站点处理数据
    # Process data for each station
    for station in stations:
        print(f"\n正在处理站点: {station}")
        
        # 加载观测数据
        # Load observation data
        obs_data = load_obs_data(station, start_time_dt, end_time_dt)
        if not obs_data:
            print(f"站点 {station} 在指定时间范围内没有观测数据，跳过")
            continue
        
        print(f"已加载观测数据，共 {len(obs_data)} 个时间点")
        
        # 加载模拟数据
        # Load simulation data
        sim_data_dict = {}
        for case in cases:
            sim_data = load_sim_data(station, case)
            sim_data_dict[case] = sim_data
        
        # 计算统计指标
        # Calculate statistics
        stats_df_dict = interpolate_and_calculate_stats(station, obs_data, sim_data_dict)
        
        # 合并结果
        # Merge results
        for var, stats_df in stats_df_dict.items():
            if var in all_stats:
                all_stats[var] = pd.concat([all_stats[var], stats_df], ignore_index=True)
            else:
                all_stats[var] = stats_df
    
    # 如果没有结果，直接退出
    # If no results, exit directly
    if not all_stats:
        print("没有找到任何有效的数据，无法计算统计指标")
        return
    
    # 计算平均统计指标
    # Calculate average statistics
    avg_stats = calculate_average_stats(all_stats)
    
    # 输出结果摘要
    # Output result summary
    print("\n统计结果摘要:")
    for var, avg_df in avg_stats.items():
        print(f"\n变量: {var}")
        # 只显示所有站点的平均结果
        # Only show average results for all stations
        print(avg_df[avg_df['station'] == 'ALL'])
    
    # 保存结果到Excel
    # Save results to Excel
    save_results_to_excel(all_stats, avg_stats)

if __name__ == "__main__":
    main() 