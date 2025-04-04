import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl.styles import PatternFill, Font

import warnings
warnings.filterwarnings("ignore")

from namelist import DATA_PATH, std_lon, std_lat
from findpoint import nearest_position

# 参数设置
# 需要处理的时间范围
start_time = '2024-04-02 00:00:00'
end_time   = '2024-04-14 00:00:00'

# 要处理的案例和变量
cases = ['metnew1', 'metnew2', 'metnew3', 'metnew4', 'cbmz']
# 观测中的变量名和模拟中对应的变量名
variables = {
    'Temperature_ground': 'Temp',
    'RH_ground': 'RH',
    'Pressure_ground': 'Pres'
}
# 模拟中存在但观测中不存在的变量
sim_only_variables = ['WS']

# 统计参数列表
stats_metrics = ['correlation', 'bias', 'rmse', 'mae', 'mean_obs', 'mean_sim', 'n_samples']

def load_data():
    """
    加载观测数据和模拟数据
    Loading observational and simulation data
    """
    print("正在加载数据...")
    
    # 读取观测数据
    obs_origin = pd.read_excel(DATA_PATH + 'OBS/obs_grd.xlsx', index_col=0)
    # 重采样为小时平均
    obs_hourly = obs_origin.resample('h').mean()
    # 选择时间范围和变量
    obs = obs_hourly.loc[start_time:end_time, list(variables.keys())]
    
    # 读取模拟数据
    sim = {}
    ds_collection = {}
    
    for case in cases:
        # 打开NC文件
        nc_path = DATA_PATH + f'mytest/postwrf/met/{case}.nc'
        ds = xr.open_dataset(nc_path)
        ds_collection[case] = ds
        
        # 获取经纬度
        lon = ds['lon']
        lat = ds['lat']
        
        # 找到最接近标准站点的格点
        nr_point = nearest_position(std_lon, std_lat, lon.values, lat.values)
        print(f"{case}: ({lon.values[nr_point]}, {lat.values[nr_point]})")
        
        # 提取该点的数据
        sim_vars = list(variables.values()) + sim_only_variables
        sim_data = ds[sim_vars].sel(south_north=nr_point[0], west_east=nr_point[1])
        
        # 转换为DataFrame
        sim[case] = sim_data.to_dataframe()
    
    return obs, sim, ds_collection

def calculate_stats(obs, sim):
    """
    计算每个变量和案例的统计指标
    Calculate statistics for each variable and case
    
    Parameters:
    -----------
    obs : DataFrame
        观测数据，index为时间
    sim : dict
        模拟数据，sim[case]为DataFrame，index为时间
    
    Returns:
    --------
    stats_results : dict
        包含不同变量的统计指标DataFrame
    """
    print("正在计算统计指标...")
    
    stats_data = []
    
    # 对每个案例和变量计算统计指标
    for case in cases:
        for obs_var, sim_var in variables.items():
            # 确保数据时间匹配
            common_times = obs.index.intersection(sim[case].index)
            if len(common_times) == 0:
                print(f"警告: {case} 和观测数据没有共同的时间点")
                continue
                
            # 选择共同时间点的数据
            obs_values = obs.loc[common_times, obs_var].values
            sim_values = sim[case].loc[common_times, sim_var].values
            
            # 过滤无效值
            valid_mask = ~np.isnan(obs_values) & ~np.isnan(sim_values)
            if np.sum(valid_mask) == 0:
                print(f"警告: {case} 的 {obs_var} 没有有效的数据点")
                continue
                
            obs_valid = obs_values[valid_mask]
            sim_valid = sim_values[valid_mask]
            
            # 计算统计指标
            correlation = np.corrcoef(obs_valid, sim_valid)[0, 1] if len(obs_valid) > 1 else np.nan
            bias = np.mean(sim_valid - obs_valid)
            rmse = np.sqrt(np.mean((sim_valid - obs_valid) ** 2))
            mae = np.mean(np.abs(sim_valid - obs_valid))
            mean_obs = np.mean(obs_valid)
            mean_sim = np.mean(sim_valid)
            n_samples = len(obs_valid)
            
            # 存储结果
            stats_data.append({
                'case': case,
                'variable': obs_var,
                'correlation': correlation,
                'bias': bias,
                'rmse': rmse,
                'mae': mae,
                'mean_obs': mean_obs,
                'mean_sim': mean_sim,
                'n_samples': n_samples
            })
    
    # 对只有模拟数据的变量，只计算mean_sim
    for case in cases:
        for var in sim_only_variables:
            if var in sim[case].columns:
                sim_values = sim[case][var].dropna().values
                if len(sim_values) > 0:
                    mean_sim = np.mean(sim_values)
                    n_samples = len(sim_values)
                    
                    # 存储结果（只有mean_sim和n_samples有效）
                    stats_data.append({
                        'case': case,
                        'variable': var,
                        'correlation': np.nan,
                        'bias': np.nan,
                        'rmse': np.nan,
                        'mae': np.nan,
                        'mean_obs': np.nan,
                        'mean_sim': mean_sim,
                        'n_samples': n_samples
                    })
    
    # 创建最终的统计结果字典，按变量组织
    stats_results = {}
    stats_df = pd.DataFrame(stats_data)
    
    # 对每个变量创建结果表格
    all_vars = list(variables.keys()) + sim_only_variables
    for var in all_vars:
        var_stats = stats_df[stats_df['variable'] == var]
        
        if not var_stats.empty:
            # 创建结果DataFrame
            result_df = pd.DataFrame(index=stats_metrics)
            
            # 添加每个案例的结果
            for case in cases:
                case_data = var_stats[var_stats['case'] == case]
                if not case_data.empty:
                    for metric in stats_metrics:
                        if metric in case_data.columns:
                            result_df.loc[metric, case] = case_data[metric].values[0]
            
            stats_results[var] = result_df
    
    return stats_results

def save_to_excel(stats_results):
    """
    将统计结果保存到Excel文件
    Save statistics results to Excel file
    
    Parameters:
    -----------
    stats_results : dict
        包含不同变量的统计指标DataFrame
    """
    # 创建输出目录
    output_path = os.path.join(DATA_PATH, '../figures/validation/stats')
    os.makedirs(output_path, exist_ok=True)
    excel_path = os.path.join(output_path, 'met_grd_stats.xlsx')
    
    # 创建ExcelWriter对象
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    
    # 保存每个变量的结果到不同的sheet
    for var, df in stats_results.items():
        df.to_excel(writer, sheet_name=var)
    
    # 保存并关闭Excel文件
    writer.close()
    
    print(f"\n统计结果已保存到: {excel_path}")

def main():
    """
    主函数
    Main function
    """
    # 加载数据
    obs, sim, ds_collection = load_data()
    
    # 计算统计指标
    stats_results = calculate_stats(obs, sim)
    
    # 输出结果摘要
    print("\n统计结果摘要:")
    for var, df in stats_results.items():
        print(f"\n变量: {var}")
        print(df)
    
    # 保存结果到Excel
    save_to_excel(stats_results)
    
    return stats_results, obs, sim

if __name__ == "__main__":
    stats_results, obs, sim = main() 