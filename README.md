# 气象和空气质量数据处理工具集

本仓库包含一系列用于气象数据和空气质量数据处理的Python工具。工具集可分为以下几个主要类别：

## WRF模拟输出数据处理工具

### ds2df.py
WRF模拟结果提取工具，支持以下三种模式：
- **profile模式**：提取单站点垂直剖面数据
- **grd模式**：提取地面网格数据
- **wyoming模式**：提取多站点数据

支持气象变量和化学组分数据处理，可导出为Excel或NetCDF格式。

### findpoint.py
空间插值工具库，包含以下功能：
- `nearest_position`：获取最临近格点坐标索引
- `nearest_positions`：获取多个最临近格点索引
- `weighted_average`：计算近邻格点的加权平均值

## 气象观测数据提取工具

### extract_gsod.py
全球气象观测数据（GSOD）提取工具，用于根据时间范围和空间范围提取气象站点数据并保存为Excel格式。主要功能：
- 支持按年份、起止日期范围筛选数据
- 支持按经纬度范围或站点ID/名称列表筛选数据
- 提供数据清洗和单位转换功能

### extract_ncdc.py
美国国家气候数据中心（NCDC）气象数据提取工具，用于提取指定条件的气象数据。主要功能：
- 支持按时间范围筛选数据
- 支持按经纬度范围或站点ID/名称列表筛选数据
- 提供数据清洗功能

### filter_stations_ncdc.py
NCDC气象站点筛选工具，用于根据经纬度范围筛选站点并查找对应的数据文件。主要功能：
- 根据经纬度范围筛选站点
- 查找站点对应的数据文件
- 比较站点表中的经纬度和数据文件中的经纬度是否一致

## 统计验证与评估工具

### met_grd_stats.py
地面网格数据统计评估工具，用于比较观测数据和模拟数据的一致性，计算各种统计指标。

### met_profile_stats.py
垂直剖面数据统计评估工具，主要功能：
- 将模拟数据插值到观测高度
- 计算各层次的统计指标（相关系数、偏差、RMSE、MAE等）
- 支持多种统计方法

### met_wyoming_stats.py
多站点高空探测数据统计评估工具，适用于处理Wyoming格式的探空数据：
- 支持多站点同时验证
- 将模拟数据插值到观测高度
- 计算各站点、各层次的统计指标

## 使用说明

各脚本文件顶部都有参数配置区域，可根据需要修改相关参数。大多数脚本可直接运行：

```bash
python3.12 脚本名.py
```

## 依赖库

- numpy
- pandas
- xarray
- scipy
- matplotlib
- dask 