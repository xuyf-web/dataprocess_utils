# 读取公共目录下的ERA5 100m UV风场和中国地面观测站点AQI数据
# 根据指定时间范围，批量绘制风场和AQI叠加图(2x2:NO2,O3,SO2,PM2.5)
# 每天生成2张图，白天(10-16)和夜间(22-04)
# 图片命名为yyyymmdd-daytime.png or yyyymmdd-nighttime.png

# Yifei Xu, 2025-04-08
# -----------------------------------------------------------

import os
import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
import logging

# Debug mode
DEBUG = False  # Set to True for detailed logging output

# 默认参数设置
OUTPUT_DIR = '.'  # 图片输出目录，默认为当前目录
DEFAULT_PLOT_DATE = '2024-08-02'
UTC_OFFSET = 8  # Beijing time is UTC+8

# 用户定义的白天和夜间时间范围
DAYTIME_START      = 10  # 白天开始时间 (小时)
DAYTIME_END        = 16    # 白天结束时间 (小时)
NIGHTTIME_START    = 22  # 夜间开始时间 (小时)
NIGHTTIME_END      = 4     # 夜间结束时间 (小时)
NIGHTTIME_NEXT_DAY = True  # 夜间结束时间是否在第二天

# 绘图参数设置 - 白天
DAYTIME_LIMITS = {
    'no2' : {'vmin': 0, 'vmax': 20, 'interval': 5},  # NO2 (ppb)
    'o3'  : {'vmin': 0, 'vmax': 100, 'interval': 20}, # O3 (ppb)
    'so2' : {'vmin': 0, 'vmax': 10, 'interval': 2},   # SO2 (ppb)
    'pm25': {'vmin': 0, 'vmax': 40, 'interval': 10}  # PM2.5 (μg/m3)
}

# 绘图参数设置 - 夜间
NIGHTTIME_LIMITS = {
    'no2' : {'vmin': 0, 'vmax': 40, 'interval': 10},  # NO2 (ppb)
    'o3'  : {'vmin': 0, 'vmax': 60, 'interval': 10}, # O3 (ppb)
    'so2' : {'vmin': 0, 'vmax': 10, 'interval': 2},   # SO2 (ppb)
    'pm25': {'vmin': 0, 'vmax': 60, 'interval': 10}  # PM2.5 (μg/m3)
}

# 污染物设置 - 名称、单位和展示位置
POLLUTANTS = {
    'no2' : {'name': 'NO$_2$', 'unit': 'ppb', 'row': 0, 'col': 0},
    'o3'  : {'name': 'O$_3$', 'unit': 'ppb', 'row': 0, 'col': 1},
    'so2' : {'name': 'SO$_2$', 'unit': 'ppb', 'row': 1, 'col': 0},
    'pm25': {'name': 'PM$_{2.5}$', 'unit': r'$\mu$g/m$^3$', 'row': 1, 'col': 1}
}

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='plot AQI and wind field combination map')
    parser.add_argument('date', nargs='?', default=DEFAULT_PLOT_DATE,
                      help='specify date (format: YYYYMMDD or YYYY-MM-DD)')
    args = parser.parse_args()
    
    # 统一日期格式为YYYY-MM-DD
    date_str = args.date.replace('-', '')
    if len(date_str) == 8:
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    
    return date_str

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 获取命令行参数中的日期
PLOT_DATE = parse_arguments()

def ensure_dir_exists(directory):
    """Ensure directory exists, create if not"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        if DEBUG:
            logging.info(f"Created output directory: {directory}")

def create_time_range(date_str, start_hour, end_hour, next_day=False, utc_offset=8):
    """创建时间范围对象
    
    返回本地时间和UTC时间范围
    """
    base_date = pd.Timestamp(date_str)
    offset = pd.Timedelta(hours=utc_offset)
    
    # 本地时间
    start_local = pd.Timestamp(f'{date_str} {start_hour:02d}:00:00')
    
    if next_day:
        next_date = (base_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        end_local = pd.Timestamp(f'{next_date} {end_hour:02d}:00:00')
    else:
        end_local = pd.Timestamp(f'{date_str} {end_hour:02d}:00:00')
    
    # UTC时间
    start_utc = start_local - offset
    end_utc = end_local - offset
    
    # 时间序列
    time_index = pd.date_range(start=start_local, end=end_local, freq='h')
    
    return {
        'start_local': start_local,
        'end_local': end_local,
        'start_utc': start_utc,
        'end_utc': end_utc,
        'time_index': time_index
    }

def load_era5_data(file_path, time_range):
    """Load ERA5 data and extract specified time range"""
    if DEBUG:
        logging.info(f"Reading ERA5 data from: {file_path}")
    era5_ds = xr.open_dataset(file_path)
    
    # Extract data within time range
    subset = era5_ds.sel(valid_time=slice(time_range['start_utc'], time_range['end_utc']))
    
    # Calculate average wind field
    u100 = subset['u100'].mean(dim='valid_time')
    v100 = subset['v100'].mean(dim='valid_time')
    
    # Extract lat/lon grid
    lon = era5_ds.longitude.values
    lat = era5_ds.latitude.values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    return u100, v100, lon_grid, lat_grid

def load_era5_data_with_next_day(file_path, next_file_path, time_range):
    """Load ERA5 data and extract specified time range, including cross-month/year cases"""
    if DEBUG:
        logging.info(f"Reading ERA5 data from multiple files:")
        logging.info(f"  - Current file: {file_path}")
        logging.info(f"  - Next file: {next_file_path}")
    
    # Read current month's data
    era5_ds = xr.open_dataset(file_path)
    subset = era5_ds.sel(valid_time=slice(time_range['start_utc'], time_range['end_utc']))
    
    # If next month's file exists, read and merge
    if os.path.exists(next_file_path):
        next_era5_ds = xr.open_dataset(next_file_path)
        next_subset = next_era5_ds.sel(valid_time=slice(time_range['start_utc'], time_range['end_utc']))
        
        # Merge datasets
        subset = xr.concat([subset, next_subset], dim='valid_time')
    
    # Calculate average wind field
    u100 = subset['u100'].mean(dim='valid_time')
    v100 = subset['v100'].mean(dim='valid_time')
    
    # Extract lat/lon grid
    lon = era5_ds.longitude.values
    lat = era5_ds.latitude.values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    return u100, v100, lon_grid, lat_grid

def load_station_data(station_file):
    """加载站点列表"""
    stations = pd.read_csv(station_file)
    stations.columns = ['stationID', 'name', 'city', 'lon', 'lat']
    return stations

def init_pollutant_dataframes(time_index, station_ids):
    """初始化存储污染物数据的DataFrames"""
    pollutants = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
    return {p: pd.DataFrame(index=time_index, columns=station_ids) for p in pollutants}

def process_aqi_data(aqi_file, daytime_index, nighttime_index):
    """Process AQI data for a single station"""
    if not os.path.exists(aqi_file):
        if DEBUG:
            logging.warning(f"AQI file not found: {aqi_file}")
        return None, None
    
    if DEBUG:
        logging.info(f"Processing AQI data from: {aqi_file}")
    
    # Read data
    aqi_data = pd.read_csv(aqi_file, sep=r'\s+', na_values='-999')
    aqi_data['datetime'] = pd.to_datetime(aqi_data['Time'], format='%Y%m%d%H')
    aqi_data = aqi_data.set_index('datetime').drop(columns=['Time'])
    
    # Unit conversion
    aqi_data['O3(ppb)'] = aqi_data['O3'] * 22.4 / 48
    aqi_data['NO2(ppb)'] = aqi_data['NO2'] * 22.4 / 46
    aqi_data['SO2(ppb)'] = aqi_data['SO2'] * 22.4 / 64
    aqi_data['CO(ppm)'] = aqi_data['CO'] * 22.4 / 28
    
    # Column mapping
    column_map = {
        'pm25': 'PM2.5',
        'pm10': 'PM10',
        'so2': 'SO2(ppb)',
        'no2': 'NO2(ppb)',
        'co': 'CO(ppm)',
        'o3': 'O3(ppb)'
    }
    
    # Extract daytime data
    day_data = {}
    for p, col in column_map.items():
        day_values = aqi_data.loc[aqi_data.index.isin(daytime_index), col]
        if not day_values.empty:
            day_data[p] = day_values
    
    # Extract nighttime data
    night_data = {}
    for p, col in column_map.items():
        night_values = aqi_data.loc[aqi_data.index.isin(nighttime_index), col]
        if not night_values.empty:
            night_data[p] = night_values
    
    return day_data, night_data

def add_shapefile_to_map(ax, sheng_reader, shiduanxian_reader):
    """向地图添加省界和十段线"""
    ax.add_feature(cfeature.ShapelyFeature(sheng_reader.geometries(), ccrs.PlateCarree(), 
                                        facecolor='none', edgecolor='k'))
    ax.add_feature(cfeature.ShapelyFeature(shiduanxian_reader.geometries(), ccrs.PlateCarree(), 
                                        facecolor='none', edgecolor='k'))
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # 添加网格线
    gl = ax.gridlines(
        xlocs=np.arange(-180, 180 + 1, 2), ylocs=np.arange(-90, 90 + 1, 2),
        draw_labels=True, x_inline=False, y_inline=False,
        linewidth=0.5, linestyle='--', color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.rotate_labels = False
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}

def create_colorbar_axes(fig, ax, is_upper_row=True):
    """创建colorbar轴"""
    pos = ax.get_position()
    
    # 根据子图位置调整colorbar位置和大小
    if is_upper_row:
        cbar_ax = fig.add_axes([pos.x0, pos.y1*0.93, pos.width*0.8, pos.y1*0.07])
    else:
        cbar_ax = fig.add_axes([pos.x0, pos.y1*0.87, pos.width*0.8, pos.y1*0.13])
    
    cbar_ax.patch.set_facecolor('white')
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks([])
    
    # 创建内部colorbar区域
    cbar_width = 0.85
    cbar_height = 0.2
    cbar_pos = cbar_ax.get_position()
    colorbar_axes = fig.add_axes([
        cbar_pos.x0 + cbar_pos.width*(1-cbar_width)/2,
        cbar_pos.y0 + cbar_pos.height*(1-cbar_height)/2, 
        cbar_pos.width*cbar_width,
        cbar_pos.height*cbar_height
    ])
    
    return colorbar_axes

def format_tick_labels(ticks):
    """格式化刻度标签，整数不显示小数点"""
    return [f"{tick:.0f}" if float(tick).is_integer() else f"{tick}" for tick in ticks]

def add_pollutant_plot(fig, ax, stations, pollutant_data, pollutant, limits, is_upper_row):
    """添加污染物散点图和colorbar"""
    # 绘制散点图
    sc = ax.scatter(stations['lon'], stations['lat'], 
                   c=pollutant_data[pollutant],
                   vmin=limits[pollutant]['vmin'], 
                   vmax=limits[pollutant]['vmax'], 
                   cmap='Spectral_r',
                   s=80, marker='o', edgecolor='k', linewidth=0.6)
    
    # 添加colorbar
    cbar_ax = create_colorbar_axes(fig, ax, is_upper_row)
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    
    # 设置colorbar标签和刻度
    cbar.set_label(f"{POLLUTANTS[pollutant]['name']} ({POLLUTANTS[pollutant]['unit']})", 
                  fontsize=14, labelpad=-55)
    
    # 刻度设置
    vmin = limits[pollutant]['vmin']
    vmax = limits[pollutant]['vmax']
    interval = limits[pollutant]['interval']
    cbar.set_ticks(np.arange(vmin, vmax+0.1, interval))
    
    # 格式化刻度标签
    tick_labels = format_tick_labels(cbar.get_ticks())
    cbar.set_ticklabels(tick_labels, fontsize=14)
    
    # 添加标题
    title = f"{POLLUTANTS[pollutant]['name']} ({POLLUTANTS[pollutant]['unit']})"
    return sc

def plot_aqi_map(u100, v100, pollutant_data, period_str, output_file, 
               is_daytime, lon_grid, lat_grid, stations, 
               sheng_reader, shiduanxian_reader):
    """Plot AQI and wind field combination map"""
    if DEBUG:
        logging.info(f"Generating {'daytime' if is_daytime else 'nighttime'} plot...")
    
    # Select plot parameters
    limits = DAYTIME_LIMITS if is_daytime else NIGHTTIME_LIMITS
    
    # Create figure
    fig = plt.figure(figsize=(14, 16), dpi=300)
    axs = fig.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Wind field arrow interval
    interval = 2
    
    # Initialize all subplots
    for i in range(2):
        for j in range(2):
            # Add base map
            add_shapefile_to_map(axs[i][j], sheng_reader, shiduanxian_reader)
            
            # Set extent and style
            axs[i][j].set_extent([116-0.01, 124, 28-0.01, 38+0.01])
            axs[i][j].tick_params(labelsize=20)
            
            # Add wind field
            qv = axs[i][j].quiver(lon_grid[::interval, ::interval], 
                               lat_grid[::interval, ::interval],
                               u100.values[::interval, ::interval], 
                               v100.values[::interval, ::interval],
                               transform=ccrs.PlateCarree(),
                               color='#2B2D2E', alpha=1, scale=100, headwidth=8)
            
            # Add wind field legend
            axs[i][j].quiverkey(qv, 0.95, -0.1, 10, '10 m/s',
                            labelpos='N', color='#2B2D2E',
                            fontproperties={'size': 12, 'weight': 'bold'})
    
    # Iterate through pollutants, add scatter plots
    for pollutant, info in POLLUTANTS.items():
        row, col = info['row'], info['col']
        
        # Add pollutant scatter plot and colorbar
        add_pollutant_plot(fig, axs[row][col], stations, pollutant_data, 
                         pollutant, limits, row == 0)
        
        # Add title
        title = f"{info['name']} ({info['unit']}) {period_str}"
        axs[row][col].set_title(title, fontsize=16, y=-0.15)
    
    # Save figure
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    if DEBUG:
        logging.info(f"Plot saved as: {output_file}")

def main():
    """Main program entry"""
    # Ensure output directory exists
    ensure_dir_exists(OUTPUT_DIR)
    
    # Base date and time
    base_date = pd.Timestamp(PLOT_DATE)
    year_str = str(base_date.year)
    month_str = f"{base_date.month:02d}"
    
    logging.info(f"Processing data for date: {PLOT_DATE}")
    
    # Data directories
    era5_uv_dir = '/data4/PUB_DATA_ALL/Reanalysis/ERA5/hourly_surface/100m_uv'
    aqi_dir = '/data4/PUB_DATA_ALL/Observation/AQI_station'
    shapefile_dir = '/data8/xuyf/data/shapefile/GS(2024)0650-SHP/'
    
    # Create time ranges
    daytime_range = create_time_range(PLOT_DATE, DAYTIME_START, DAYTIME_END, 
                                    False, UTC_OFFSET)
    nighttime_range = create_time_range(PLOT_DATE, NIGHTTIME_START, NIGHTTIME_END, 
                                      NIGHTTIME_NEXT_DAY, UTC_OFFSET)
    
    # Load station information
    station_file = os.path.join(aqi_dir, 'stationlist.csv')
    stations = load_station_data(station_file)
    
    # Get next day's date information (for cross-month/year)
    next_day_date = base_date + pd.Timedelta(days=1)
    next_year_str = str(next_day_date.year)
    next_month_str = f"{next_day_date.month:02d}"
    
    # Prepare ERA5 file paths
    current_era5_file = os.path.join(era5_uv_dir, f'ERA5_hourly_100m_uv_{base_date.year}-{month_str}.nc')
    next_era5_file = os.path.join(era5_uv_dir, f'ERA5_hourly_100m_uv_{next_day_date.year}-{next_month_str}.nc')
    
    # Load ERA5 data
    # Daytime data (no cross-month/year needed)
    u100_day, v100_day, lon_grid, lat_grid = load_era5_data(current_era5_file, daytime_range)
    
    # Nighttime data (may need cross-month/year)
    if NIGHTTIME_NEXT_DAY and (next_day_date.year != base_date.year or next_day_date.month != base_date.month):
        u100_night, v100_night, _, _ = load_era5_data_with_next_day(current_era5_file, next_era5_file, nighttime_range)
    else:
        u100_night, v100_night, _, _ = load_era5_data(current_era5_file, nighttime_range)
    
    # Initialize dataframes
    day_pollutants = init_pollutant_dataframes(daytime_range['time_index'], stations['stationID'])
    night_pollutants = init_pollutant_dataframes(nighttime_range['time_index'], stations['stationID'])
    
    # Process each station's data
    for station_id in stations['stationID']:
        # Process daytime data (current date file only)
        current_aqi_file = os.path.join(aqi_dir, f'{year_str}/AQI_{station_id}_{year_str}.txt')
        
        # Check if cross-month or cross-year is needed
        next_day_needed = False
        next_day_file = ""
        
        if NIGHTTIME_NEXT_DAY:
            # Get next day's date
            next_day_date = base_date + pd.Timedelta(days=1)
            next_day_year = str(next_day_date.year)
            
            # If next day is different year or month, need to read another file
            if next_day_date.year != base_date.year or next_day_date.month != base_date.month:
                next_day_needed = True
                next_day_file = os.path.join(aqi_dir, f'{next_day_year}/AQI_{station_id}_{next_day_year}.txt')
        
        # Process current date's AQI data
        current_day_data, current_night_data = process_aqi_data(
            current_aqi_file, daytime_range['time_index'], nighttime_range['time_index'])
        
        # If cross-month/year needed and file exists, process next day's data
        next_day_data = None
        if next_day_needed and os.path.exists(next_day_file):
            _, next_day_data = process_aqi_data(
                next_day_file, [], nighttime_range['time_index'])
        
        # Merge current date and next day's nighttime data
        night_data = current_night_data
        if next_day_data:
            if night_data is None:
                night_data = next_day_data
            else:
                # Merge data from both dictionaries
                for pollutant, values in next_day_data.items():
                    if pollutant not in night_data:
                        night_data[pollutant] = values
                    else:
                        # Merge two Series
                        night_data[pollutant].update(values)
        
        # Fill daytime data
        if current_day_data:
            for pollutant, series in current_day_data.items():
                for timestamp, value in series.items():
                    day_pollutants[pollutant].loc[timestamp, station_id] = value
        
        # Fill nighttime data
        if night_data:
            for pollutant, series in night_data.items():
                for timestamp, value in series.items():
                    night_pollutants[pollutant].loc[timestamp, station_id] = value
    
    logging.info("Data processing completed")
    
    # Calculate averages
    day_avg = {p: df.mean() for p, df in day_pollutants.items()}
    night_avg = {p: df.mean() for p, df in night_pollutants.items()}
    
    # Load shapefiles
    sheng = Reader(os.path.join(shapefile_dir, 'sheng.shp'))
    shiduanxian = Reader(os.path.join(shapefile_dir, 'shiduanxian.shp'))
    
    # Generate daytime image
    day_period = f"{DAYTIME_START:02d}-{DAYTIME_END:02d}:00LT {base_date.strftime('%Y%m%d')}"
    day_output = os.path.join(OUTPUT_DIR, f"{base_date.strftime('%Y%m%d')}-daytime.png")
    
    plot_aqi_map(
        u100_day, v100_day, day_avg, day_period, day_output, 
        True, lon_grid, lat_grid, stations, sheng, shiduanxian
    )
    
    # Generate nighttime image
    night_period = f"{NIGHTTIME_START:02d}-{NIGHTTIME_END:02d}:00LT {base_date.strftime('%Y%m%d')}"
    night_output = os.path.join(OUTPUT_DIR, f"{base_date.strftime('%Y%m%d')}-nighttime.png")
    
    plot_aqi_map(
        u100_night, v100_night, night_avg, night_period, night_output,
        False, lon_grid, lat_grid, stations, sheng, shiduanxian
    )
    
    logging.info(f"Plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

