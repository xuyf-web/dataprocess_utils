import shutil
import netCDF4 as nc
import numpy as np
import pandas as pd
import os
import glob
import sys
from datetime import datetime, timedelta
from contextlib import contextmanager


@contextmanager
def open_nc_file(filepath, mode='r'):
    """安全打开netCDF文件的上下文管理器，确保文件正确关闭"""
    f = None
    try:
        f = nc.Dataset(filepath, mode=mode, format='NETCDF4')
        yield f
    finally:
        if f is not None:
            f.close()


def ll_area(lat, res):
    """计算网格面积（km²）"""
    Re = 6371.392  # 地球平均半径(km)
    X = Re * np.cos(lat * (np.pi / 180)) * (np.pi / 180) * res
    Y = Re * (np.pi / 180) * res
    return X * Y


def meic2wrf(lon_inp, lat_inp, lon, lat, emis):
    """
    双线性插值：将MEIC网格数据插值到WRF网格
    """
    ox = lat[0]
    oy = lon[0]
    dx_meic = lat[1] - lat[0] if len(lat) > 1 else 0.25
    dy_meic = lon[1] - lon[0] if len(lon) > 1 else 0.25

    def inp(ix, iy, dx, dy, cdx, cdy):
        ix = min(max(ix, 0), len(lat) - 2)
        iy = min(max(iy, 0), len(lon) - 2)
        return (emis[ix, iy] * cdx * cdy +
                emis[ix, iy + 1] * cdx * dy +
                emis[ix + 1, iy + 1] * dx * dy +
                emis[ix + 1, iy] * dx * cdy)

    def std_p(p, o, res):
        if res == 0:
            res = 0.25
        p_rel = (p - o) / res
        ip = int(np.floor(p_rel))
        dp = p_rel - ip
        cdp = 1 - dp
        return dp, cdp, ip

    def inp_p(px, py):
        dx, cdx, ix = std_p(px, ox, dx_meic)
        dy, cdy, iy = std_p(py, oy, dy_meic)
        return inp(ix, iy, dx, dy, cdx, cdy)

    emis_inp = np.zeros(lon_inp.shape, dtype='float32')
    for y in range(lat_inp.shape[0]):
        for x in range(lon_inp.shape[1]):
            emis_inp[y, x] = inp_p(lat_inp[y, x], lon_inp[y, x])

    return emis_inp


def extend_vertical_profile(zfac_orig, target_levels, decay_rate=0.7):
    """将垂直分配系数从原始层数扩展到目标层数"""
    n_sectors, n_original = zfac_orig.shape
    
    if target_levels <= n_original:
        return zfac_orig[:, :target_levels]
    
    zfac_extended = np.zeros((n_sectors, target_levels))
    
    for s in range(n_sectors):
        zfac_extended[s, :n_original] = zfac_orig[s, :]
        last_value = zfac_orig[s, -1]
        for z in range(n_original, target_levels):
            zfac_extended[s, z] = last_value * (decay_rate ** (z - n_original + 1))
    
    for s in range(n_sectors):
        total = np.sum(zfac_extended[s, :])
        if total > 0:
            zfac_extended[s, :] = zfac_extended[s, :] / total * np.sum(zfac_orig[s, :])
    
    return zfac_extended


def sec2zt(sec, zfac, tfac, target_z_levels):
    """时空分配系数计算"""
    n_sectors, n_original_levels = zfac.shape
    n_hours = tfac.shape[1]
    lat, lon = sec.shape

    result = np.zeros((n_hours, target_z_levels, lat, lon), dtype=np.float32)
    zfac_extended = extend_vertical_profile(zfac, target_z_levels)
    
    for s in range(n_sectors):
        for t in range(n_hours):
            for z in range(target_z_levels):
                result[t, z, :, :] += sec * tfac[s, t] * zfac_extended[s, z]

    return result


def add_wrf42_global_attributes(nc_file, wrfinput_file, start_time):
    """添加WRF4.2标准的全局属性"""
    with open_nc_file(wrfinput_file, 'r') as f_inp:
        nc_file.setncattr('TITLE', 'WRF-Chem Emission File (MEIC Inventory)')
        nc_file.setncattr('WRF_VERSION', 'WRF v4.2')
        nc_file.setncattr('MODEL_CONFIGURATION', 'chem')
        nc_file.setncattr('SIMULATION_START_DATE', start_time.strftime('%Y-%m-%d_%H:%M:%S'))

        for attr in ['DX', 'DY', 'MAP_PROJ', 'TRUELAT1', 'TRUELAT2',
                     'STAND_LON', 'POLE_LAT', 'POLE_LON', 'GMT', 'JULYR', 'JULDAY']:
            if hasattr(f_inp, attr):
                nc_file.setncattr(attr, getattr(f_inp, attr))

        nc_file.setncattr('EMISSION_SOURCE', 'MEIC 2023 Inventory')
        nc_file.setncattr('EMISSION_CHEM_SCHEME', 'CB05 (emiss_opt=14)')
        nc_file.setncattr('CREATION_DATE', datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        nc_file.setncattr('UNITS_NOTES', 'Gases: mol km^-2 hr^-1; Aerosols: ug m^-2 s^-1')


def read_meic_gas_ton(filepath, sectors, lon, lat, lon_inp, lat_inp, 
                      sec_z_d, sec_t_d, target_z_levels, resdata, M):
    """
    读取MEIC气体数据（单位：吨/网格/月）
    用于：CO, NH3, NOx, SO2
    输出：mol km⁻² hr⁻¹
    """
    with open_nc_file(filepath) as f_post:
        meic_lon = f_post.variables['lon'][:]
        meic_lat = f_post.variables['lat'][:]
        
        meic_area_km2 = ll_area(meic_lat, resdata)
        meic_area_2d = meic_area_km2[:, np.newaxis]
        
        section_data = []
        for sec in sectors:
            if sec in f_post.variables:
                ton_per_grid_month = f_post.variables[sec][:, :]
                # 吨 → 克：×1e6，克 → 摩尔：÷M
                mol_per_grid_month = ton_per_grid_month * 1e6 / M
                mol_per_km2_month = mol_per_grid_month / meic_area_2d
                mol_per_km2_month_wrf = meic2wrf(lon_inp, lat_inp, meic_lon, meic_lat, mol_per_km2_month)
                data_final = mol_per_km2_month_wrf / (30 * 24)
                section_data.append(data_final)
            else:
                section_data.append(np.zeros_like(lon_inp))
    
    sections = [sec2zt(emis, sec_z_d, sec_t_d, target_z_levels) for emis in section_data]
    return sum(sections)


def read_meic_voc_mmol(filepath, sectors, lon, lat, lon_inp, lat_inp,
                       sec_z_d, sec_t_d, target_z_levels, resdata):
    """
    读取MEIC VOC机制物种数据（单位：百万摩尔/网格/月）
    用于：XYL, TOL, ETH, OLE, PAR, FORM, ALD2, ALDX, ISOP, TERP, 
         MEOH, ETOH, IOLE, CH4, ETHA, NVOL, UNR
    输出：mol km⁻² hr⁻¹
    """
    with open_nc_file(filepath) as f_post:
        meic_lon = f_post.variables['lon'][:]
        meic_lat = f_post.variables['lat'][:]
        
        meic_area_km2 = ll_area(meic_lat, resdata)
        meic_area_2d = meic_area_km2[:, np.newaxis]
        
        section_data = []
        for sec in sectors:
            if sec in f_post.variables:
                # 读取：百万摩尔/网格/月
                mmol_per_grid_month = f_post.variables[sec][:, :]
                # 百万摩尔 → 摩尔：× 1e6
                mol_per_grid_month = mmol_per_grid_month * 1e6
                mol_per_km2_month = mol_per_grid_month / meic_area_2d
                mol_per_km2_month_wrf = meic2wrf(lon_inp, lat_inp, meic_lon, meic_lat, mol_per_km2_month)
                data_final = mol_per_km2_month_wrf / (30 * 24)
                section_data.append(data_final)
            else:
                section_data.append(np.zeros_like(lon_inp))
    
    sections = [sec2zt(emis, sec_z_d, sec_t_d, target_z_levels) for emis in section_data]
    return sum(sections)


def read_meic_aerosol_data(filepath, sectors, lon, lat, lon_inp, lat_inp,
                           sec_z_d, sec_t_d, target_z_levels, resdata):
    """
    读取MEIC气溶胶数据（单位：吨/网格/月）
    用于：BC, OC, PM25, PM10
    输出：μg m⁻² s⁻¹
    """
    with open_nc_file(filepath) as f_post:
        meic_lon = f_post.variables['lon'][:]
        meic_lat = f_post.variables['lat'][:]
        
        meic_area_km2 = ll_area(meic_lat, resdata)
        meic_area_m2 = meic_area_km2 * 1e6
        meic_area_m2_2d = meic_area_m2[:, np.newaxis]
        
        section_data = []
        for sec in sectors:
            if sec in f_post.variables:
                ton_per_grid_month = f_post.variables[sec][:, :]
                # 吨/网格/月 → 吨/m²/月
                ton_per_m2_month = ton_per_grid_month / meic_area_m2_2d
                ton_per_m2_month_wrf = meic2wrf(lon_inp, lat_inp, meic_lon, meic_lat, ton_per_m2_month)
                # 吨/m²/月 → μg/m²/s
                ug_per_m2_s = ton_per_m2_month_wrf * 1e12 / (30 * 24 * 3600)
                section_data.append(ug_per_m2_s)
            else:
                section_data.append(np.zeros_like(lon_inp))
    
    sections = [sec2zt(emis, sec_z_d, sec_t_d, target_z_levels) for emis in section_data]
    return sum(sections)


def itp_dis(ent_inp, meicdir, run_folder, sec_z_d, sec_t_d, resdata, start_date_str, fangda):
    """主插值函数：处理MEIC数据并生成WRF-Chem输入文件"""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

    with open_nc_file(ent_inp) as f_inp:
        lon_inp = f_inp.variables['XLONG'][0, :]
        lat_inp = f_inp.variables['XLAT'][0, :]

        if 'bottom_top' in f_inp.dimensions:
            target_z_levels = len(f_inp.dimensions['bottom_top'])
        else:
            target_z_levels = 34

    print(f"检测到气象文件垂直层数：{target_z_levels}")

    sectors = ['act', 'idt', 'pwr', 'rdt', 'tpt']
    meic_data = {}
    
    # ==================== 读取经纬度信息（用于面积计算）====================
    # 从第一个文件获取经纬度
    first_file = os.path.join(meicdir, 'CO.nc')
    if not os.path.exists(first_file):
        print(f"错误：找不到文件 {first_file}")
        return
    
    with open_nc_file(first_file) as f_sample:
        lon = f_sample.variables['lon'][:]
        lat = f_sample.variables['lat'][:]
    
    # ==================== 1. 无机气体（吨/网格/月，需要分子量）====================
    inorganic_specs = {'CO': 28, 'NH3': 17, 'NOx': 46, 'SO2': 64}
    for spec, M in inorganic_specs.items():
        filepath = os.path.join(meicdir, f'{spec}.nc')
        if os.path.exists(filepath):
            print(f"读取 {spec}.nc...")
            meic_data[spec] = read_meic_gas_ton(
                filepath, sectors, lon, lat, lon_inp, lat_inp,
                sec_z_d, sec_t_d, target_z_levels, resdata, M
            )
        else:
            print(f"警告：{spec}.nc 不存在")
            meic_data[spec] = None
    
    # ==================== 2. VOC机制物种（百万摩尔/网格/月，不需要分子量）====================
    voc_mmol_specs = [
        'XYL', 'TOL', 'ETH', 'OLE', 'PAR', 'FORM', 'ALD2', 'ALDX',
        'ISOP', 'TERP', 'MEOH', 'ETOH', 'IOLE', 'CH4', 'ETHA', 'NVOL', 'UNR'
    ]
    for spec in voc_mmol_specs:
        filepath = os.path.join(meicdir, f'{spec}.nc')
        if os.path.exists(filepath):
            print(f"读取 {spec}.nc...")
            meic_data[spec] = read_meic_voc_mmol(
                filepath, sectors, lon, lat, lon_inp, lat_inp,
                sec_z_d, sec_t_d, target_z_levels, resdata
            )
        else:
            print(f"警告：{spec}.nc 不存在")
            meic_data[spec] = None
    
    # ==================== 3. 颗粒物（吨/网格/月）====================
    aerosol_specs = ['BC', 'OC', 'PM25', 'PM10']
    for spec in aerosol_specs:
        filepath = os.path.join(meicdir, f'{spec}.nc')
        if os.path.exists(filepath):
            print(f"读取 {spec}.nc...")
            meic_data[spec] = read_meic_aerosol_data(
                filepath, sectors, lon, lat, lon_inp, lat_inp,
                sec_z_d, sec_t_d, target_z_levels, resdata
            )
        else:
            print(f"警告：{spec}.nc 不存在")
            meic_data[spec] = None
    
    # 应用放大系数
    for key in meic_data:
        if meic_data[key] is not None:
            meic_data[key] = meic_data[key] * fangda
    
    # ==================== 4. CB05化学方案需要的40个物种 ====================
    cb05_spec_names = [
        # 气相无机物种 (6个)
        'E_CO', 'E_NH3', 'E_NO', 'E_NO2', 'E_SO2', 'E_HCL',
        # 气相有机物种 (19个)
        'E_ISO', 'E_TERP', 'E_TOL', 'E_XYL', 'E_ETH', 'E_CSL',
        'E_HCHO', 'E_ALD', 'E_ALDX', 'E_HC3', 'E_HC5', 'E_HC8',
        'E_OLT', 'E_OLI', 'E_OL2', 'E_KET', 'E_ORA2', 'E_CH3OH', 'E_C2H5OH',
        # 气溶胶 - PM2.5爱根核模态 (5个)
        'E_PM25I', 'E_ECI', 'E_ORGI', 'E_SO4I', 'E_NO3I',
        # 气溶胶 - PM2.5积聚模态 (5个)
        'E_PM25J', 'E_ECJ', 'E_ORGJ', 'E_SO4J', 'E_NO3J',
        # 气溶胶 - PM10粗模态 (5个)
        'E_SO4C', 'E_NO3C', 'E_ORGC', 'E_ECC', 'E_PM10'
    ]
    
    n_species = len(cb05_spec_names)
    wrf_spec_emis = [np.zeros((12, target_z_levels) + lon_inp.shape, dtype=np.float32) 
                      for _ in range(n_species)]
    
    # ==================== 5. 无机气体分配 ====================
    if meic_data.get('CO') is not None:
        wrf_spec_emis[cb05_spec_names.index('E_CO')] = meic_data['CO']
    
    if meic_data.get('NH3') is not None:
        wrf_spec_emis[cb05_spec_names.index('E_NH3')] = meic_data['NH3']
    
    if meic_data.get('SO2') is not None:
        wrf_spec_emis[cb05_spec_names.index('E_SO2')] = meic_data['SO2']
    
    # NO和NO2分配（假设NOx中90%为NO，10%为NO2）
    if meic_data.get('NOx') is not None:
        wrf_spec_emis[cb05_spec_names.index('E_NO')] = meic_data['NOx'] * 0.9
        wrf_spec_emis[cb05_spec_names.index('E_NO2')] = meic_data['NOx'] * 0.1
    
    # HCl：如果没有数据，使用SO2的5%估算
    if meic_data.get('SO2') is not None:
        wrf_spec_emis[cb05_spec_names.index('E_HCL')] = meic_data['SO2'] * 0.05
    
    # ==================== 6. VOC物种分配 ====================
    # 使用MEIC的VOC机制物种数据直接赋值或估算
    
    # 直接对应的物种
    direct_mapping = {
        'ISOP': 'E_ISO',
        'TERP': 'E_TERP',
        'TOL': 'E_TOL',
        'XYL': 'E_XYL',
        'ETH': 'E_ETH',
        'FORM': 'E_HCHO',
        'ALD2': 'E_ALD',
        'ALDX': 'E_ALDX',
        'MEOH': 'E_CH3OH',
        'ETOH': 'E_C2H5OH',
    }
    
    for meic_spec, cb05_spec in direct_mapping.items():
        if meic_data.get(meic_spec) is not None:
            wrf_spec_emis[cb05_spec_names.index(cb05_spec)] = meic_data[meic_spec]
    
    # 处理OLE和IOLE → OLT, OLI, OL2
    if meic_data.get('OLE') is not None and meic_data.get('IOLE') is not None:
        total_ole = meic_data['OLE'] + meic_data['IOLE']
        wrf_spec_emis[cb05_spec_names.index('E_OLT')] = total_ole * 0.4
        wrf_spec_emis[cb05_spec_names.index('E_OLI')] = total_ole * 0.4
        wrf_spec_emis[cb05_spec_names.index('E_OL2')] = total_ole * 0.2
    elif meic_data.get('OLE') is not None:
        wrf_spec_emis[cb05_spec_names.index('E_OLT')] = meic_data['OLE'] * 0.5
        wrf_spec_emis[cb05_spec_names.index('E_OLI')] = meic_data['OLE'] * 0.3
        wrf_spec_emis[cb05_spec_names.index('E_OL2')] = meic_data['OLE'] * 0.2
    
    # 处理PAR → HC3, HC5, HC8
    if meic_data.get('PAR') is not None:
        wrf_spec_emis[cb05_spec_names.index('E_HC3')] = meic_data['PAR'] * 0.4
        wrf_spec_emis[cb05_spec_names.index('E_HC5')] = meic_data['PAR'] * 0.35
        wrf_spec_emis[cb05_spec_names.index('E_HC8')] = meic_data['PAR'] * 0.25
    
    # 处理KET（酮类）：使用NVOL或估算
    if meic_data.get('NVOL') is not None:
        wrf_spec_emis[cb05_spec_names.index('E_KET')] = meic_data['NVOL'] * 0.5
    
    # 处理ORA2（乙酸）：使用UNR或估算
    if meic_data.get('UNR') is not None:
        wrf_spec_emis[cb05_spec_names.index('E_ORA2')] = meic_data['UNR'] * 0.3
    
    # 处理CSL（甲酚）：如果没有数据，使用TOL的5%估算
    if meic_data.get('TOL') is not None:
        wrf_spec_emis[cb05_spec_names.index('E_CSL')] = meic_data['TOL'] * 0.05
    
    # 如果某些物种缺失，使用总VOC的分配系数估算
    # 计算总VOC（用于估算缺失物种）
    total_voc = None
    voc_list = ['XYL', 'TOL', 'ETH', 'OLE', 'PAR', 'FORM', 'ALD2', 'ALDX',
                'ISOP', 'TERP', 'MEOH', 'ETOH', 'IOLE', 'CH4', 'ETHA', 'NVOL', 'UNR']
    for spec in voc_list:
        if meic_data.get(spec) is not None:
            if total_voc is None:
                total_voc = meic_data[spec].copy()
            else:
                total_voc = total_voc + meic_data[spec]
    
    # 估算缺失物种的分配系数
    if total_voc is not None:
        # 检查并估算缺失的物种
        missing_estimates = {
            'E_CSL': 0.01,      # 甲酚
            'E_KET': 0.05,      # 酮类
            'E_ORA2': 0.02,     # 乙酸
        }
        for cb05_spec, factor in missing_estimates.items():
            idx = cb05_spec_names.index(cb05_spec)
            current = wrf_spec_emis[idx]
            if np.max(current) == 0:
                wrf_spec_emis[idx] = total_voc * factor
    
    # ==================== 7. 气溶胶物种分配 ====================
    bc_emis = meic_data.get('BC', np.zeros_like(wrf_spec_emis[0]))
    oc_emis = meic_data.get('OC', np.zeros_like(wrf_spec_emis[0]))
    pm25_emis = meic_data.get('PM25', np.zeros_like(wrf_spec_emis[0]))
    pm10_emis = meic_data.get('PM10', np.zeros_like(wrf_spec_emis[0])) 
    
        # ========== 添加调试输出 ==========
    print("=" * 60)
    print("气溶胶排放数据统计:")
    print(f"BC 最大值: {np.max(bc_emis):.6f}")
    print(f"OC 最大值: {np.max(oc_emis):.6f}")
    print(f"PM25 最大值: {np.max(pm25_emis):.6f}")
    print(f"PM10 最大值: {np.max(pm10_emis):.6f}")
    print(f"PM10/PM25 比例: {np.max(pm10_emis)/np.max(pm25_emis):.4f}" if np.max(pm25_emis) > 0 else "PM25为0")
    print("=" * 60)
    # ========== 调试输出结束 ========== 
   
    
    # PM2.5质量分配
    wrf_spec_emis[cb05_spec_names.index('E_PM25I')] = pm25_emis * 0.2
    wrf_spec_emis[cb05_spec_names.index('E_PM25J')] = pm25_emis * 0.8
    
    # EC分配
    wrf_spec_emis[cb05_spec_names.index('E_ECI')] = bc_emis * 0.4
    wrf_spec_emis[cb05_spec_names.index('E_ECJ')] = bc_emis * 0.6
    wrf_spec_emis[cb05_spec_names.index('E_ECC')] = bc_emis * 0.2
    
    # OC分配
    wrf_spec_emis[cb05_spec_names.index('E_ORGI')] = oc_emis * 0.3
    wrf_spec_emis[cb05_spec_names.index('E_ORGJ')] = oc_emis * 0.7
    wrf_spec_emis[cb05_spec_names.index('E_ORGC')] = oc_emis * 0.2
    
    # 硫酸盐和硝酸盐分配（假设PM2.5中硫酸盐占20%，硝酸盐占30%）
    wrf_spec_emis[cb05_spec_names.index('E_SO4I')] = pm25_emis * 0.2 * 0.3
    wrf_spec_emis[cb05_spec_names.index('E_SO4J')] = pm25_emis * 0.2 * 0.7
    wrf_spec_emis[cb05_spec_names.index('E_NO3I')] = pm25_emis * 0.3 * 0.3
    wrf_spec_emis[cb05_spec_names.index('E_NO3J')] = pm25_emis * 0.3 * 0.7
    
    # 粗模态（PM10-PM2.5）
    coarse_mass = pm10_emis - pm25_emis
    coarse_mass = np.maximum(coarse_mass, 0)
    
    wrf_spec_emis[cb05_spec_names.index('E_SO4C')] = coarse_mass * 0.2
    wrf_spec_emis[cb05_spec_names.index('E_NO3C')] = coarse_mass * 0.15
    wrf_spec_emis[cb05_spec_names.index('E_PM10')] = pm10_emis
    
    print(f"成功创建 {len(wrf_spec_emis)} 个CB05排放物种")
    
    # ==================== 8. 生成WRF-Chem输入文件 ====================
    domain_name = 'd01'
    for d in ['d01', 'd02', 'd03']:
        if d in ent_inp:
            domain_name = d
            break

    for ihour in [0, 12]:
        output_filename = f'wrfchemi_{ihour:02d}z_{domain_name}'
        output_file = os.path.join(run_folder, output_filename)

        if os.path.exists(output_file):
            os.remove(output_file)

        with open_nc_file(output_file, 'w') as f_chem:
            # 创建维度
            f_chem.createDimension('Time', None)
            f_chem.createDimension('emissions_zdim', target_z_levels)
            f_chem.createDimension('south_north', lon_inp.shape[0])
            f_chem.createDimension('west_east', lon_inp.shape[1])
            f_chem.createDimension('DateStrLen', 19)

            current_start_time = start_date + timedelta(hours=ihour)
            add_wrf42_global_attributes(f_chem, ent_inp, current_start_time)

            # 时间变量
            times_var = f_chem.createVariable('Times', 'S1', ('Time', 'DateStrLen'))
            times_var.setncattr('long_name', 'Time of emission')

            time_list = []
            for t in range(12):
                current_time = start_date + timedelta(hours=ihour + t)
                time_str = current_time.strftime('%Y-%m-%d_%H:%M:%S')
                time_list.append(list(time_str))
            times_var[:] = np.array(time_list, dtype='S1')

            # 经纬度
            lon_var = f_chem.createVariable('XLONG', 'f4', ('south_north', 'west_east'))
            lon_var.setncattr('long_name', 'Longitude')
            lon_var.setncattr('units', 'degrees_east')
            lon_var[:] = lon_inp

            lat_var = f_chem.createVariable('XLAT', 'f4', ('south_north', 'west_east'))
            lat_var.setncattr('long_name', 'Latitude')
            lat_var.setncattr('units', 'degrees_north')
            lat_var[:] = lat_inp

            # 创建变量
            for i, spec in enumerate(cb05_spec_names):
                # 判断是气相还是气溶胶
                if spec in ['E_CO', 'E_NH3', 'E_NO', 'E_NO2', 'E_SO2', 'E_HCL'] + \
                   [f'E_{x}' for x in ['ISO', 'TERP', 'TOL', 'XYL', 'ETH', 'CSL',
                                        'HCHO', 'ALD', 'ALDX', 'HC3', 'HC5', 'HC8',
                                        'OLT', 'OLI', 'OL2', 'KET', 'ORA2', 'CH3OH', 'C2H5OH']]:
                    var = f_chem.createVariable(spec, 'f4', ('Time', 'emissions_zdim', 'south_north', 'west_east'))
                    var.units = 'mol km^-2 hr^-1'
                else:
                    var = f_chem.createVariable(spec, 'f4', ('Time', 'emissions_zdim', 'south_north', 'west_east'))
                    var.units = 'ug m^-2 s^-1'
                var.description = 'EMISSIONS'

            # 写入数据
            for i, spec in enumerate(cb05_spec_names):
                if spec in f_chem.variables:
                    data_slice = wrf_spec_emis[i][ihour:ihour + 12, :, :, :]
                    f_chem.variables[spec][:] = data_slice.astype(np.float32)

        print(f"成功生成文件: {output_file}")


def get_wrfinput_files(run_path):
    """获取所有wrfinput文件"""
    if not os.path.isdir(run_path):
        return []
    pattern = os.path.join(run_path, "wrfinput_d*")
    return sorted(glob.glob(pattern))


if __name__ == '__main__':
    WRFHOME = os.environ.get('WRFHOME')
    if WRFHOME is None:
        print("❌ 错误：WRFHOME环境变量未设置！")
        sys.exit(1)
    
    run_folder = f'{WRFHOME}/WRF/run'

    if len(sys.argv) != 3:
        print("用法：python meictowrf.py 起报日期 预报时长")
        sys.exit(1)
    
    START_DATE = sys.argv[1]
    try:
        datetime.strptime(START_DATE, "%Y-%m-%d")
    except ValueError:
        print(f"错误：日期格式错误")
        sys.exit(1)

    # 读取分配系数
    try:
        sec = pd.read_excel('meictowrf.xlsx')
        fangda = sec['fangda'][0]
        resdata = 0.25
        sec_z_d = np.array([sec['agr_z_d'][0:11], sec['ind_z_d'][0:11],
                            sec['pow_z_d'][0:11], sec['res_z_d'][0:11],
                            sec['tra_z_d'][0:11]])
        sec_t_d = np.array([sec['agr_t_d'], sec['ind_t_d'], sec['pow_t_d'],
                            sec['res_t_d'], sec['tra_t_d']])
        print(f"成功读取垂直分配系数：{sec_z_d.shape}")
    except Exception as e:
        print(f"读取Excel文件失败：{e}")
        sys.exit(1)

    month = START_DATE[5:7]
    meicdir = f'{WRFHOME}/peizhishuju/MEIC2023/MEIC{month}/'
    print(f'正在分配MEIC{month}给WRF')

    if not os.path.isdir(meicdir):
        print(f"错误：MEIC数据目录 {meicdir} 不存在")
        sys.exit(1)

    ent_inputs = get_wrfinput_files(run_folder)
    if not ent_inputs:
        print("错误：未找到wrfinput文件")
        sys.exit(1)

    print(f"找到WRF输入文件：{[os.path.basename(f) for f in ent_inputs]}")

    for ent_inp in ent_inputs:
        try:
            itp_dis(ent_inp, meicdir, run_folder, sec_z_d, sec_t_d, resdata, START_DATE, fangda)
        except Exception as e:
            print(f"处理文件 {ent_inp} 失败：{e}")
            import traceback
            traceback.print_exc()
            continue

    print("所有文件处理完成！")
