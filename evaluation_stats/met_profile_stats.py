import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from openpyxl.styles import PatternFill, Font

import warnings
warnings.filterwarnings("ignore")

from namelist import DATA_PATH, std_lon, std_lat, profile_periods

# 参数设置
# 需要处理的时间范围
start_time = '2024-04-02 00:00:00'
end_time   = '2024-04-14 00:00:00'

# 要处理的案例和变量
cases = ['metnew1', 'metnew2', 'metnew3', 'metnew4', 'cbmz']
variables = ['Temp', 'RH', 'Pres', 'WS']  # 不包括Height，它将用于插值
zlevels = [29, 29, 29, 29, 29]  # 每个case对应的垂直层数限制

# 统计参数列表
stats_metrics = ['correlation', 'bias', 'rmse', 'mae', 'mean_obs', 'mean_sim', 'n_samples']

# 统计方法选择
# 1: 数据拼接后计算统计参数 (默认)
# 2: 分别计算参数后平均
# 3: 输出所有方法的结果
method_choice = 1

def load_data():
    """
    加载观测数据和模拟数据
    Loading observational and simulation data
    """
    print("正在加载数据...")
    
    # 读取观测数据
    obs_origin = pd.read_excel(DATA_PATH + 'OBS/obs_air.xlsx', index_col=0)
    met_names = ['Temperaturer_airship', 'RHr_airship', 'Pressurer_airship',
                'WSr_airship', 'Height_Meteor_airship_height']
    obs = obs_origin.loc[start_time:end_time, met_names]
    obs.columns = ['Temp', 'RH', 'Pres', 'WS', 'Height']
    
    # 单位转换：将观测气压从kPa转换为hPa (乘以10)
    # Unit conversion: convert observed pressure from kPa to hPa (multiply by 10)
    obs['Pres'] = obs['Pres'] * 10.0
    print(f"气压单位转换：观测数据气压范围 {obs['Pres'].min():.2f}-{obs['Pres'].max():.2f} hPa")
    
    # 读取模拟数据
    sim = {}
    for i, case in enumerate(cases):
        sim[case] = {}
        for var in variables + ['Height']:  # 添加Height用于插值
            excelpath = DATA_PATH + f'mytest/postwrf/met/{case}.xlsx'
            if var == 'Height':  # 高度数据需要特殊处理
                height = pd.read_excel(excelpath, sheet_name=var, index_col=0)
                data_z40 = height.values
                data_z39 = 0.5 * (data_z40[:,1:] + data_z40[:,:-1])
                sim[case][var] = pd.DataFrame(data_z39, index=height.index, 
                                            columns=height.columns[:-1]).loc[:, :zlevels[i]]
            else:
                sim[case][var] = pd.read_excel(excelpath, sheet_name=var, 
                                            index_col=0).loc[:, :zlevels[i]]
    
    # 显示模拟数据气压范围
    for case in cases:
        pres_min = sim[case]['Pres'].min().min()
        pres_max = sim[case]['Pres'].max().max()
        print(f"模拟数据({case})气压范围: {pres_min:.2f}-{pres_max:.2f} hPa")
    
    return obs, sim

def interpolate_and_calculate_stats(obs, sim):
    """
    将sim中的变量按照高度插值到obs的高度，然后计算统计指标
    Interpolate variables in sim to obs heights, then calculate statistics
    
    Parameters:
    -----------
    obs : DataFrame
        观测数据，index为时间，列包含多个变量和Height
    sim : dict
        模拟数据，sim[case][var]为DataFrame，index为时间，列为不同高度序号
    
    Returns:
    --------
    stats_results : dict
        包含不同变量的统计指标DataFrame，按照要求的格式组织
    interpolated_sim : dict
        插值后的模拟数据，可用于后续分析
    """
    print("正在进行插值和统计计算...")
    
    # 创建结果存储结构
    stats_data = []  # 用于存储Method 2的结果
    profile_stats_data = []  # 用于存储每条廓线的统计结果（Method 1）
    interpolated_sim = {}
    
    # 时间键列表，用于选择有观测和模拟数据的时间点
    timekeys = [
        '2024-04-02 10:00', '2024-04-02 15:00',
        '2024-04-03 10:00',
        '2024-04-04 09:00', '2024-04-04 17:00',
        '2024-04-05 08:00',
        '2024-04-08 09:00',
        '2024-04-12 18:00', '2024-04-12 21:00',
        '2024-04-13 07:00',
    ]
    
    # 对每个case和变量进行处理
    for case in cases:
        interpolated_sim[case] = {}
        
        for var in variables:
            # 创建存储插值结果的DataFrame
            interpolated_values = []
            observed_values = []
            valid_times = []
            
            # 存储每条廓线的统计结果
            profile_stats = []
            
            # 遍历每个时间点
            for timekey in timekeys:
                # 检查模拟数据中是否有该时间点
                if timekey not in sim[case][var].index:
                    continue
                
                # 获取观测时间段
                if timekey in profile_periods:
                    period = profile_periods[timekey]
                    
                    # 检查观测数据中是否有该时间段
                    if period.start not in obs.index or period.stop not in obs.index:
                        continue
                    
                    # 获取该时间段内的观测数据
                    period_obs = obs.loc[period]
                    if period_obs.empty:
                        continue
                    
                    # 获取模拟高度和变量值
                    sim_heights = sim[case]['Height'].loc[timekey].values
                    sim_values = sim[case][var].loc[timekey].values
                    
                    # 获取观测高度和变量值
                    obs_heights = period_obs['Height'].values
                    obs_values = period_obs[var].values
                    
                    # 创建插值函数
                    try:
                        f = interpolate.interp1d(
                            sim_heights, 
                            sim_values, 
                            kind='linear', 
                            bounds_error=False, 
                            fill_value='extrapolate'
                        )
                        
                        # 在观测高度上进行插值
                        interpolated_value = f(obs_heights)
                        
                        # 存储结果
                        interpolated_values.append(interpolated_value)
                        observed_values.append(obs_values)
                        valid_times.append(timekey)
                        
                        # 计算该廓线的统计指标（Method 1）
                        profile_correlation = np.corrcoef(obs_values, interpolated_value)[0, 1]
                        profile_bias = np.mean(interpolated_value - obs_values)
                        profile_rmse = np.sqrt(np.mean((interpolated_value - obs_values) ** 2))
                        profile_mae = np.mean(np.abs(interpolated_value - obs_values))
                        profile_mean_obs = np.mean(obs_values)
                        profile_mean_sim = np.mean(interpolated_value)
                        profile_n_samples = len(obs_values)
                        
                        # 存储该廓线的统计结果
                        profile_stats.append({
                            'time': timekey,
                            'correlation': profile_correlation,
                            'bias': profile_bias,
                            'rmse': profile_rmse,
                            'mae': profile_mae,
                            'mean_obs': profile_mean_obs,
                            'mean_sim': profile_mean_sim,
                            'n_samples': profile_n_samples
                        })
                        
                    except Exception as e:
                        print(f"插值错误 ({case}, {var}, {timekey}): {e}")
            
            # 如果没有有效数据，跳过
            if not valid_times:
                continue
            
            # 存储插值结果
            interpolated_sim[case][var] = {
                'times': valid_times,
                'interp_values': interpolated_values,
                'obs_values': observed_values
            }
            
            # 计算整体统计指标（Method 2）
            flat_obs = np.concatenate([arr for arr in observed_values])
            flat_sim = np.concatenate([arr for arr in interpolated_values])
            
            correlation = np.corrcoef(flat_obs, flat_sim)[0, 1] if len(flat_obs) > 1 else np.nan
            bias = np.mean(flat_sim - flat_obs)
            rmse = np.sqrt(np.mean((flat_sim - flat_obs) ** 2))
            mae = np.mean(np.abs(flat_sim - flat_obs))
            mean_obs = np.mean(flat_obs)
            mean_sim = np.mean(flat_sim)
            n_samples = len(flat_obs)
            
            # 存储整体统计结果
            stats_data.append({
                'case': case,
                'variable': var,
                'correlation': correlation,
                'bias': bias,
                'rmse': rmse,
                'mae': mae,
                'mean_obs': mean_obs,
                'mean_sim': mean_sim,
                'n_samples': n_samples
            })
            
            # 计算每条廓线统计结果的平均值（Method 1的平均）
            profile_stats_df = pd.DataFrame(profile_stats)
            profile_mean_stats = {
                'case': case,
                'variable': var,
                'correlation': profile_stats_df['correlation'].mean(),
                'bias': profile_stats_df['bias'].mean(),
                'rmse': profile_stats_df['rmse'].mean(),
                'mae': profile_stats_df['mae'].mean(),
                'mean_obs': profile_stats_df['mean_obs'].mean(),
                'mean_sim': profile_stats_df['mean_sim'].mean(),
                'n_samples': profile_stats_df['n_samples'].sum()
            }
            profile_stats_data.append(profile_mean_stats)
    
    # 创建最终的统计结果字典，按变量组织
    stats_results = {}
    stats_df = pd.DataFrame(stats_data)
    profile_stats_df = pd.DataFrame(profile_stats_data)
    
    # 根据要求格式组织结果
    for var in variables:
        # 创建多级索引的DataFrame
        var_stats = stats_df[stats_df['variable'] == var]
        var_profile_stats = profile_stats_df[profile_stats_df['variable'] == var]
        
        if not var_stats.empty:
            # 根据用户选择的方法创建结果DataFrame
            if method_choice == 1:  # 数据拼接后计算统计参数
                # 创建结果DataFrame
                result_df = pd.DataFrame(index=pd.MultiIndex.from_product([
                    ['整体统计'],
                    stats_metrics
                ], names=['统计方法', '统计参数']))
                
                # 添加Method 2的结果
                for case in cases:
                    case_data = var_stats[var_stats['case'] == case]
                    if not case_data.empty:
                        for metric in stats_metrics:
                            result_df.loc[('整体统计', metric), case] = case_data[metric].values[0]
                
                # 添加说明行
                result_df.loc[('说明', '说明'), '说明'] = '整体统计结果（所有数据点合并计算）'
                
            elif method_choice == 2:  # 分别计算参数后平均
                # 创建结果DataFrame
                result_df = pd.DataFrame(index=pd.MultiIndex.from_product([
                    ['廓线平均'],
                    stats_metrics
                ], names=['统计方法', '统计参数']))
                
                # 添加Method 1的平均结果
                for case in cases:
                    case_data = var_profile_stats[var_profile_stats['case'] == case]
                    if not case_data.empty:
                        for metric in stats_metrics:
                            result_df.loc[('廓线平均', metric), case] = case_data[metric].mean()
                
                # 添加说明行
                result_df.loc[('说明', '说明'), '说明'] = '廓线平均结果（先计算每条廓线的统计参数再取平均）'
                
            else:  # 输出所有方法的结果
                # 创建结果DataFrame
                result_df = pd.DataFrame(index=pd.MultiIndex.from_product([
                    ['整体统计', '廓线平均'],
                    stats_metrics
                ], names=['统计方法', '统计参数']))
                
                # 添加Method 2的结果
                for case in cases:
                    case_data = var_stats[var_stats['case'] == case]
                    if not case_data.empty:
                        for metric in stats_metrics:
                            result_df.loc[('整体统计', metric), case] = case_data[metric].values[0]
                
                # 添加Method 1的平均结果
                for case in cases:
                    case_data = var_profile_stats[var_profile_stats['case'] == case]
                    if not case_data.empty:
                        for metric in stats_metrics:
                            result_df.loc[('廓线平均', metric), case] = case_data[metric].mean()
                
                # 添加说明行
                result_df.loc[('说明', '说明'), '说明'] = '上部分为整体统计结果（所有数据点合并计算），下部分为廓线平均结果（先计算每条廓线的统计参数再取平均）'
            
            stats_results[var] = result_df
    
    return stats_results, interpolated_sim

def main():
    """
    主函数
    Main function
    """
    # 加载数据
    obs, sim = load_data()
    
    # 执行插值和统计计算
    stats_results, interpolated_sim = interpolate_and_calculate_stats(obs, sim)
    
    # 输出结果
    print("\n统计结果摘要:")
    for var, df in stats_results.items():
        print(f"\n变量: {var}")
        print(df)
    
    # 保存结果到Excel
    output_path = os.path.join(DATA_PATH, '../figures/validation/stats')
    os.makedirs(output_path, exist_ok=True)
    excel_path = os.path.join(output_path, 'met_profile_stats.xlsx')
    
    # 创建ExcelWriter对象
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    
    # 保存每个变量的结果到不同的sheet
    for var, df in stats_results.items():
        # 重置索引，将多级索引转换为列
        df_reset = df.reset_index()
        df_reset.to_excel(writer, sheet_name=var, index=False)
    
    # 保存并关闭Excel文件
    writer.close()
    
    print(f"\n统计结果已保存到: {excel_path}")
    
    return stats_results, interpolated_sim

if __name__ == "__main__":
    stats_results, interpolated_sim = main() 