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

### meic2wrf/
MEIC 排放清单到 WRF-Chem 排放输入的处理工具（面向 MEIC 2023、WRF v4.2、CB05 机制 `emiss_opt=14`）：
- `merged.py`：将按部门拆分的 MEIC 原始文件合并为按物种分类的文件
- `meictowrf.py`：把 MEIC 数据整理并插值到 WRF-Chem 网格，生成 `wrfchemi_*` 排放文件

详见 `meic2wrf/README.md`。

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

## 绘图工具

### plot_mixed_font/mixed_font.py
中文 / 英文 / 数学公式混排工具。解决 matplotlib 中「中文 + `$...$` 公式」无法同框的问题：
只要字符串含 `$`，整串都会被 mathtext 数学引擎渲染，中文会因数学字体缺少中文字形而变成豆腐块。
本工具按 `$...$` 自动拆分文本，普通文本段走常规路径（中文正常）、公式段走 mathtext，再用
`HPacker` 水平拼接居中。

```python
from mixed_font import setup_fonts, mixed_text
setup_fonts()                       # 西文 Times New Roman、中文 SimSun、公式 stix
mixed_text(ax, 0.5, 0.5, r'Times New Roman和宋体: $e^{i \pi} + 1 = 0$', fontsize=28)
```

依赖 Times New Roman、SimSun 字体（已放在 `~/.local/share/fonts/`）。新增字体后若 matplotlib
找不到，删除字体缓存重建即可：`rm ~/.cache/matplotlib/fontlist-*.json`。直接运行该脚本会生成 demo 图。

### plotAQI/
空气质量（AQI）逐日绘图工具：
- `plot_oneday.py`：绘制单日 AQI 空间分布（含昼/夜）
- `batch_plot.sh`：按日期批量调用绘图

### polar_subplots_lits.py
基于 cartopy / rasterio 的极地投影多子图绘图脚本，用于读取 TIFF（如夏季 LST）栅格并叠加矢量边界出图。

### TangColorScheme.py
唐风配色方案集合（`color_schemes`，5 组各 5 色），供绘图时统一取色。

## 其它工具

### init_project.sh
项目脚手架脚本，按当前项目的目录结构快速创建新项目文件夹：`./init_project.sh <project_name>`。

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