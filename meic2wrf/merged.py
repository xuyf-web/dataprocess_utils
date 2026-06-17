#!/usr/bin/env python3
"""
MEIC排放清单合并工具
功能：将按部门拆分的MEIC原始文件合并为按物种分类的文件
输入：~/pythonliu/WRFchem_tool/MEIC/MEIC2023M/2023_MM_部门_物种.nc
输出：~/pythonliu/WRFchem_tool/MEIC/MEIC2023_merged/MEICMM/物种.nc
单位：保留原始单位，清晰标注
"""

import os
import glob
import numpy as np
import netCDF4 as nc
from datetime import datetime

# ==================== 配置参数 ====================
# 输入输出路径

WRFHOME = os.environ.get('WRFHOME')
    if WRFHOME is None:
        print("❌ 错误：WRFHOME环境变量未设置！")
        print("请执行：export WRFHOME=/mnt/ssd-swap/WRFHOME")
        sys.exit(1)
INPUT_BASE = f"{WRFHOME}/peizhishuju/MEIC2023M"
OUTPUT_BASE = f"{WRFHOME}/peizhishuju/MEIC2023_merged"

# 部门名称与缩写映射（保持与原始代码一致）
SECTOR_MAPPING = {
    'agriculture': 'act',
    'industry': 'idt',
    'power': 'pwr',
    'residential': 'rdt',
    'transportation': 'tpt'
}

# MEIC网格信息
LAT_MIN, LAT_MAX = 10.125, 60.0  # 纬度范围
LON_MIN, LON_MAX = 70.125, 150.0  # 经度范围
RESOLUTION = 0.25  # 分辨率
LAT_N = 200  # 纬度格点数
LON_N = 320  # 经度格点数

# ==================== 核心函数 ====================

def get_month_days(month):
    """返回月份的天数"""
    month_days = {
        '01': 31, '02': 28, '03': 31, '04': 30,
        '05': 31, '06': 30, '07': 31, '08': 31,
        '09': 30, '10': 31, '11': 30, '12': 31
    }
    return month_days.get(month, 30)

def parse_filename(filename):
    """
    解析MEIC文件名
    返回: (month, sector, species, is_cb05)
    示例: 2023_01_agriculture_PM2.5.nc -> ('01', 'agriculture', 'PM2.5', False)
          2023_01_industry_CB05_XYL.nc -> ('01', 'industry', 'XYL', True)
    """
    basename = os.path.basename(filename)
    parts = basename.replace('.nc', '').split('_')
    
    # 格式: 2023_MM_部门_[CB05_]物种
    month = parts[1]  # 第2部分是月份
    sector = parts[2]  # 第3部分是部门
    
    # 判断是否为CB05机制物种
    if 'CB05' in parts:
        is_cb05 = True
        # CB05后的部分是物种名
        cb05_index = parts.index('CB05')
        species = parts[cb05_index + 1]
    else:
        is_cb05 = False
        species = parts[-1]  # 最后一部分是物种名
    
    return month, sector, species, is_cb05

def group_files_by_species(all_files):
    """
    将所有文件按物种分组
    返回: {
        'PM2.5': {'agriculture': path1, 'industry': path2, ...},
        'SO2': {...},
        ...
    }
    """
    species_dict = {}
    
    for file_path in all_files:
        try:
            month, sector, species, is_cb05 = parse_filename(file_path)
            
            # 如果是CB05物种，在物种名后标记（可选，根据需要）
            # if is_cb05:
            #     species = f"CB05_{species}"
            
            if species not in species_dict:
                species_dict[species] = {}
            
            species_dict[species][sector] = file_path
            
        except Exception as e:
            print(f"⚠️ 跳过文件 {file_path}: {e}")
    
    return species_dict

def read_sector_data(file_path, expected_shape=(200, 320)):
    """
    读取单个部门文件的数据
    返回: (data, lon, lat)
    """
    with nc.Dataset(file_path, 'r') as f:
        # 读取数据
        data = f.variables['z'][:].astype(np.float32)
        
        # 确保数据形状正确
        if data.shape != expected_shape:
            if data.size == expected_shape[0] * expected_shape[1]:
                data = data.reshape(expected_shape)
            else:
                raise ValueError(f"数据形状 {data.shape} 与期望形状 {expected_shape} 不匹配")
        
        # 反转纬度方向（MEIC数据纬度是从北到南，WRF通常是从南到北）
        data = data[::-1, :]
        
        # 将负值设为0
        data = np.where(data > 0, data, 0.0)
        
        # 读取经纬度（如果存在）
        if 'lon' in f.variables:
            lon = f.variables['lon'][:].astype(np.float32)
        else:
            lon = np.arange(LON_MIN, LON_MAX, RESOLUTION, dtype=np.float32)
            
        if 'lat' in f.variables:
            lat = f.variables['lat'][:].astype(np.float32)
        else:
            lat = np.arange(LAT_MIN, LAT_MAX, RESOLUTION, dtype=np.float32)
    
    return data, lon, lat

def write_merged_file(species_name, sector_data, output_path, lon, lat, month):
    """
    写入合并后的物种文件
    """
    with nc.Dataset(output_path, 'w', format='NETCDF4') as f_out:
        # 创建维度
        f_out.createDimension('lon', len(lon))
        f_out.createDimension('lat', len(lat))
        
        # 写入经度
        lon_var = f_out.createVariable('lon', np.float32, ('lon',))
        lon_var[:] = lon
        lon_var.units = 'degrees_east'
        lon_var.long_name = 'longitude'
        lon_var.standard_name = 'longitude'
        
        # 写入纬度
        lat_var = f_out.createVariable('lat', np.float32, ('lat',))
        lat_var[:] = lat
        lat_var.units = 'degrees_north'
        lat_var.long_name = 'latitude'
        lat_var.standard_name = 'latitude'
        
        # 判断物种类型，设置单位
        mass_species = ['PM2.5', 'PM25','PM10', 'BC', 'OC', 'POC', 'EC']
        gas_species = ['SO2', 'NOx', 'CO', 'NH3', 'NMVOC']
        
        
        if species_name in mass_species:
            units = 'metric_ton month-1 gridcell-1'
            description = f'{species_name} mass emissions per grid cell per month'
        elif species_name in gas_species:
            units = 'million_mol month-1 gridcell-1'
            description = f'{species_name} emissions per grid cell per month (10^6 moles)'
        else:
        # CB05或其他机制物种，可能是摩尔单位
            units = 'million_mol month-1 gridcell-1'
            description = f'{species_name} (CB05 mechanism) emissions per grid cell per month (10^6 moles)'
        
        
        # 写入各部门数据
        for sector_abbr, data in sector_data.items():
            var = f_out.createVariable(sector_abbr, np.float32, ('lat', 'lon'))
            var[:] = data
            
            # 添加变量属性
            var.units = units
            var.description = description
            var.sector = sector_abbr
            var.long_name = f'{species_name} from {sector_abbr} sector'
        
        # 添加全局属性
        f_out.title = f'MEIC 2023 {species_name} emissions'
        f_out.institution = 'MEIC (Multi-resolution Emission Inventory for China)'
        f_out.source = 'MEIC 2023 monthly emissions'
        f_out.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        f_out.references = 'http://meicmodel.org'
        f_out.comment = f'Merged by sector into single species file. Original units: {units}'
        f_out.month = month
        f_out.grid_resolution = f'{RESOLUTION} degree'
        f_out.grid_extent = f'lon: {LON_MIN}-{LON_MAX}, lat: {LAT_MIN}-{LAT_MAX}'

def process_month(month):
    """
    处理单个月份的所有物种
    """
    print(f"\n{'='*50}")
    print(f"处理月份: {month}")
    
    # 输入目录
    input_dir = os.path.expanduser(INPUT_BASE)
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 匹配该月份的所有文件
    pattern = os.path.join(input_dir, f"2023_{month}_*.nc")
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"❌ 未找到 {month} 月的文件")
        return
    
    print(f"找到 {len(all_files)} 个文件")
    
    # 按物种分组
    species_dict = group_files_by_species(all_files)
    print(f"识别出 {len(species_dict)} 个物种: {list(species_dict.keys())}")
    
    # 创建输出目录
    output_dir = os.path.join(os.path.expanduser(OUTPUT_BASE), f"MEIC{month}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个物种
    for species_name, sector_files in species_dict.items():
        print(f"\n▶ 处理物种: {species_name}")
        
        sector_data = {}
        ref_lon, ref_lat = None, None
        
        # 读取每个部门的数据
        for sector, file_path in sector_files.items():
            if sector not in SECTOR_MAPPING:
                print(f"  ⚠️ 跳过未知部门: {sector}")
                continue
            
            try:
                print(f"  读取: {os.path.basename(file_path)}")
                data, lon, lat = read_sector_data(file_path)
                
                sector_abbr = SECTOR_MAPPING[sector]
                sector_data[sector_abbr] = data
                
                if ref_lon is None:
                    ref_lon, ref_lat = lon, lat
                    
            except Exception as e:
                print(f"  ❌ 读取失败: {e}")
                continue
        
        # 写入合并文件
        if sector_data:
            output_file = os.path.join(output_dir, f"{species_name}.nc")
            write_merged_file(species_name, sector_data, output_file, 
                            ref_lon, ref_lat, month)
            print(f"  ✅ 已保存: {output_file}")
            
            # 可选：输出统计信息
            total_by_sector = {s: np.sum(d) for s, d in sector_data.items()}
            total_all = sum(total_by_sector.values())
            print(f"     总量: {total_all:.2f} (单位: 按物种类型)")
        else:
            print(f"  ❌ 无有效数据")

def main():
    """主函数"""
    print("="*60)
    print("MEIC排放清单合并工具")
    print("="*60)
    
    # 处理所有月份
    months = [f"{i:02d}" for i in range(1, 13)]
    
    for month in months:
        process_month(month)
    
    print("\n" + "="*60)
    print("✅ 所有月份处理完成！")
    print(f"输出目录: {os.path.expanduser(OUTPUT_BASE)}")
    print("="*60)

if __name__ == "__main__":
    main()
