# MEIC 到 WRF-Chem 排放文件处理工具

本项目用于把 MEIC 排放清单数据整理并插值到 WRF-Chem 模式网格，生成 WRF-Chem 可读取的 `wrfchemi_*` 排放输入文件。当前代码主要面向 MEIC 2023 排放清单、WRF v4.2、CB05 化学机制（`emiss_opt=14`）的处理流程。

## 项目目的

WRF-Chem 运行需要与模拟区域、网格、垂直层和时间步相匹配的排放输入文件。MEIC 原始清单通常按月份、部门、物种存储在规则经纬度网格上，不能直接被 WRF-Chem 使用。本项目完成以下转换：

1. 将按部门拆分的 MEIC 原始 NetCDF 文件合并为按物种组织的文件。
2. 将 MEIC 经纬度网格排放插值到 WRF 网格。
3. 按部门应用时间分配系数和垂直分配系数。
4. 将 MEIC 物种映射或拆分为 WRF-Chem CB05 机制需要的排放变量。
5. 生成 `wrfchemi_00z_d*` 和 `wrfchemi_12z_d*` 文件，供 WRF-Chem 读取。

## 文件结构

```text
MEIC/
├── merged.py          # MEIC 原始部门文件合并工具
├── meictowrf.py       # MEIC 到 WRF-Chem wrfchemi 文件主转换脚本
├── meictowrf.py.bak   # meictowrf.py 的备份版本
└── README.md          # 项目说明
```

运行 `meictowrf.py` 时还需要一个当前目录下的 `meictowrf.xlsx`，用于提供部门时间分配、垂直分配和放大系数。该文件当前不在项目目录中，需要运行前自行准备。

## 总体流程

推荐流程分为两步：

1. 数据预处理：使用 `merged.py` 将 MEIC 原始文件从“月份-部门-物种”格式合并为“月份-物种”格式。
2. WRF-Chem 转换：使用 `meictowrf.py` 读取合并后的 MEIC 数据、WRF `wrfinput_d*` 文件和 `meictowrf.xlsx` 分配系数，输出 `wrfchemi_*` 文件。

数据流如下：

```text
MEIC 原始文件
  2023_MM_agriculture_*.nc
  2023_MM_industry_*.nc
  2023_MM_power_*.nc
  2023_MM_residential_*.nc
  2023_MM_transportation_*.nc
        |
        | merged.py
        v
按物种合并的 MEIC 月排放文件
  MEICMM/CO.nc, NOx.nc, SO2.nc, PM25.nc, ...
        |
        | meictowrf.py + wrfinput_d* + meictowrf.xlsx
        v
WRF-Chem 排放输入文件
  wrfchemi_00z_d01, wrfchemi_12z_d01, ...
```

## 核心功能

### `merged.py`

`merged.py` 的目标是把 MEIC 原始按部门拆分的文件合并成按物种组织的 NetCDF 文件。它默认读取：

```text
$WRFHOME/peizhishuju/MEIC2023M/2023_MM_部门_物种.nc
```

并输出：

```text
$WRFHOME/peizhishuju/MEIC2023_merged/MEICMM/物种.nc
```

部门名称会转换为脚本内部使用的 5 个缩写：

| 原始部门 | 合并后变量名 |
| --- | --- |
| `agriculture` | `act` |
| `industry` | `idt` |
| `power` | `pwr` |
| `residential` | `rdt` |
| `transportation` | `tpt` |

关键函数：

- `parse_filename(filename)`：解析 MEIC 原始文件名，识别月份、部门、物种和是否为 CB05 物种。
- `group_files_by_species(all_files)`：按物种把同一个月的各部门文件分组。
- `read_sector_data(file_path)`：读取单部门文件中的 `z` 变量，修正形状，反转纬度方向，并将负值置零。
- `write_merged_file(...)`：写出合并后的物种文件，包含 `lon`、`lat` 和各部门变量。
- `process_month(month)`：处理单个月份的全部物种。
- `main()`：循环处理 1-12 月。

注意：当前 `merged.py` 在 `WRFHOME` 检查处存在缩进错误，并且使用了 `sys.exit()` 但没有导入 `sys`。运行该脚本前需要先修正这些问题。另外，`merged.py` 默认输出目录是 `MEIC2023_merged`，而 `meictowrf.py` 默认读取目录是 `MEIC2023`，两者路径需要统一，例如修改其中一个路径，或将合并结果复制/链接到 `MEIC2023/MEICMM/`。

### `meictowrf.py`

`meictowrf.py` 是主转换脚本。它会读取 WRF 网格、MEIC 月排放数据和分配系数，输出 WRF-Chem 排放文件。

默认输入：

```text
$WRFHOME/WRF/run/wrfinput_d*
$WRFHOME/peizhishuju/MEIC2023/MEICMM/*.nc
当前运行目录/meictowrf.xlsx
```

默认输出：

```text
$WRFHOME/WRF/run/wrfchemi_00z_d*
$WRFHOME/WRF/run/wrfchemi_12z_d*
```

关键函数：

- `open_nc_file(filepath, mode='r')`：NetCDF 文件上下文管理器，确保文件关闭。
- `ll_area(lat, res)`：按纬度和分辨率计算经纬度网格面积，单位为 `km2`。
- `meic2wrf(lon_inp, lat_inp, lon, lat, emis)`：把 MEIC 规则经纬度网格上的排放双线性插值到 WRF 网格。
- `extend_vertical_profile(zfac_orig, target_levels, decay_rate=0.7)`：当 WRF 垂直层数超过 Excel 中的 11 层垂直分配系数时，按衰减率向上扩展并保持总分配量一致。
- `sec2zt(sec, zfac, tfac, target_z_levels)`：把二维月排放场扩展为 `Time x emissions_zdim x south_north x west_east` 的四维排放场。
- `add_wrf42_global_attributes(...)`：从 `wrfinput` 复制关键投影和网格属性，并写入 WRF-Chem 排放文件全局属性。
- `read_meic_gas_ton(...)`：读取 `CO`、`NH3`、`NOx`、`SO2` 等气体物种，并转换为 `mol km^-2 hr^-1`。
- `read_meic_voc_mmol(...)`：读取 CB05 VOC 机制物种，并转换为 `mol km^-2 hr^-1`。
- `read_meic_aerosol_data(...)`：读取 `BC`、`OC`、`PM25`、`PM10`，并转换为 `ug m^-2 s^-1`。
- `itp_dis(...)`：主处理函数，完成读取、插值、时间/垂直分配、物种映射、NetCDF 写出。
- `get_wrfinput_files(run_path)`：查找 `$WRFHOME/WRF/run/` 下的所有 `wrfinput_d*` 文件。

## 物种处理逻辑

`meictowrf.py` 生成 CB05 机制使用的一组 `E_*` 排放变量，主要包括：

- 无机气体：`E_CO`、`E_NH3`、`E_NO`、`E_NO2`、`E_SO2`、`E_HCL`
- VOC：`E_ISO`、`E_TERP`、`E_TOL`、`E_XYL`、`E_ETH`、`E_HCHO`、`E_ALD`、`E_ALDX`、`E_HC3`、`E_HC5`、`E_HC8`、`E_OLT`、`E_OLI`、`E_OL2` 等
- 气溶胶：`E_PM25I`、`E_PM25J`、`E_ECI`、`E_ECJ`、`E_ORGI`、`E_ORGJ`、`E_SO4I`、`E_SO4J`、`E_NO3I`、`E_NO3J`、`E_PM10` 等

重要分配假设：

- `NOx` 按 `90% NO + 10% NO2` 分配。
- 没有独立 `HCL` 数据时，使用 `SO2 * 0.05` 估算。
- `OLE` 和 `IOLE` 会拆分为 `OLT`、`OLI`、`OL2`。
- `PAR` 会拆分为 `HC3`、`HC5`、`HC8`。
- `PM25` 按 `20%` 爱根核模态和 `80%` 积聚模态分配。
- `BC`、`OC`、硫酸盐、硝酸盐、粗模态颗粒物按脚本内固定比例估算。
- 若部分 VOC 派生物种缺失，会用总 VOC 的固定比例估算。

## 运行前准备

### 1. Python 环境

建议使用 conda 创建独立环境：

```bash
conda create -n meic-wrf python=3.10 -y
conda activate meic-wrf
conda install -c conda-forge numpy pandas netcdf4 openpyxl -y
```

依赖说明：

- `numpy`：数组计算
- `pandas`：读取 `meictowrf.xlsx`
- `netCDF4`：读取和写入 NetCDF 文件
- `openpyxl`：让 `pandas.read_excel()` 能读取 `.xlsx`

### 2. 设置 `WRFHOME`

两个脚本都依赖环境变量 `WRFHOME`。Linux/macOS 示例：

```bash
export WRFHOME=/mnt/ssd-swap/WRFHOME
```

PowerShell 示例：

```powershell
$env:WRFHOME = "D:\path\to\WRFHOME"
```

脚本会从该目录推导输入输出路径，例如：

```text
$WRFHOME/WRF/run/
$WRFHOME/peizhishuju/MEIC2023/
$WRFHOME/peizhishuju/MEIC2023M/
```

### 3. 准备 WRF 输入文件

确保 WRF 已完成 `real.exe` 或等价前处理步骤，并在以下目录存在至少一个 `wrfinput_d*` 文件：

```text
$WRFHOME/WRF/run/wrfinput_d01
```

如果有嵌套区域，也可以存在：

```text
$WRFHOME/WRF/run/wrfinput_d02
$WRFHOME/WRF/run/wrfinput_d03
```

脚本会自动识别所有 `wrfinput_d*`，并为每个 domain 分别生成 `wrfchemi_*_d*`。

### 4. 准备 MEIC 数据

`meictowrf.py` 按起报日期中的月份查找数据，例如起报日期为 `2023-01-15` 时，会读取：

```text
$WRFHOME/peizhishuju/MEIC2023/MEIC01/
```

该目录下至少应包含：

```text
CO.nc
NH3.nc
NOx.nc
SO2.nc
BC.nc
OC.nc
PM25.nc
PM10.nc
```

如需完整 VOC 物种映射，还应包含：

```text
XYL.nc
TOL.nc
ETH.nc
OLE.nc
PAR.nc
FORM.nc
ALD2.nc
ALDX.nc
ISOP.nc
TERP.nc
MEOH.nc
ETOH.nc
IOLE.nc
CH4.nc
ETHA.nc
NVOL.nc
UNR.nc
```

每个文件需要包含：

```text
lon
lat
act
idt
pwr
rdt
tpt
```

其中 `act/idt/pwr/rdt/tpt` 分别代表农业、工业、电力、居民和交通部门排放。

### 5. 准备 `meictowrf.xlsx`

运行 `meictowrf.py` 的当前目录必须有 `meictowrf.xlsx`。脚本会读取以下列：

```text
fangda
agr_z_d, ind_z_d, pow_z_d, res_z_d, tra_z_d
agr_t_d, ind_t_d, pow_t_d, res_t_d, tra_t_d
```

含义：

- `fangda`：所有物种统一乘上的放大系数。
- `*_z_d`：不同部门的垂直分配系数。脚本当前读取前 11 行，如果 WRF 垂直层更多，会自动向上扩展。
- `*_t_d`：不同部门的时间分配系数。脚本根据这些列生成小时排放。通常需要提供 24 个小时分配值，因为输出会切成 `00z` 和 `12z` 两个 12 小时文件。

部门缩写对应关系：

| Excel 前缀 | 部门 |
| --- | --- |
| `agr` | agriculture / 农业 |
| `ind` | industry / 工业 |
| `pow` | power / 电力 |
| `res` | residential / 居民 |
| `tra` | transportation / 交通 |

## 可自定义内容

常用可修改项：

- `meictowrf.xlsx` 中的 `fangda`：统一调整排放强度。
- `meictowrf.xlsx` 中的时间分配系数：控制每日小时排放变化。
- `meictowrf.xlsx` 中的垂直分配系数：控制排放在 WRF 垂直层中的分布。
- `meictowrf.py` 中的 `resdata`：MEIC 网格分辨率，当前为 `0.25` 度。
- `meictowrf.py` 中的 `meicdir`：MEIC 按月份数据目录。
- `meictowrf.py` 中的 `inorganic_specs`、`voc_mmol_specs`、`aerosol_specs`：要读取的 MEIC 物种列表。
- `meictowrf.py` 中的 `direct_mapping` 和后续分配比例：MEIC 到 CB05 物种的映射和估算假设。
- `merged.py` 中的 `INPUT_BASE` 和 `OUTPUT_BASE`：原始 MEIC 文件和合并输出目录。

## 如何运行

### 方式 A：已有合并后的 MEIC 物种文件

如果已经准备好 `$WRFHOME/peizhishuju/MEIC2023/MEICMM/*.nc`，可以直接运行主转换脚本：

```bash
cd /path/to/MEIC
conda activate meic-wrf
export WRFHOME=/mnt/ssd-swap/WRFHOME
python meictowrf.py 2023-01-01 24
```

PowerShell 示例：

```powershell
cd D:\Download\MEIC
conda activate meic-wrf
$env:WRFHOME = "D:\path\to\WRFHOME"
python meictowrf.py 2023-01-01 24
```

参数说明：

```text
python meictowrf.py 起报日期 预报时长
```

- `起报日期`：格式必须为 `YYYY-MM-DD`，脚本会用其中的月份决定读取 `MEICMM` 目录。
- `预报时长`：当前代码只检查该参数是否存在，实际没有使用该值控制输出长度。脚本固定生成 `00z` 和 `12z` 两个 12 小时文件。

### 方式 B：从 MEIC 原始部门文件开始

先修正 `merged.py` 的缩进和 `sys` 导入问题，并确认输入目录中有类似以下命名的原始文件：

```text
$WRFHOME/peizhishuju/MEIC2023M/2023_01_agriculture_PM2.5.nc
$WRFHOME/peizhishuju/MEIC2023M/2023_01_industry_SO2.nc
$WRFHOME/peizhishuju/MEIC2023M/2023_01_power_CB05_XYL.nc
```

然后运行：

```bash
cd /path/to/MEIC
conda activate meic-wrf
export WRFHOME=/mnt/ssd-swap/WRFHOME
python merged.py
```

合并完成后，将输出目录与 `meictowrf.py` 的读取目录保持一致。可选做法包括：

```bash
cp -r $WRFHOME/peizhishuju/MEIC2023_merged/MEIC* $WRFHOME/peizhishuju/MEIC2023/
```

之后再运行：

```bash
python meictowrf.py 2023-01-01 24
```

## 期望产出

对每个 `wrfinput_d*`，脚本会在 `$WRFHOME/WRF/run/` 下生成：

```text
wrfchemi_00z_d01
wrfchemi_12z_d01
```

如果存在 `wrfinput_d02` 或 `wrfinput_d03`，还会生成：

```text
wrfchemi_00z_d02
wrfchemi_12z_d02
wrfchemi_00z_d03
wrfchemi_12z_d03
```

每个输出文件包含：

- `Times`：12 个逐小时时间戳。
- `XLONG`、`XLAT`：WRF 网格经纬度。
- `emissions_zdim`：与 `wrfinput` 中 `bottom_top` 维度一致的垂直层数。
- 多个 `E_*` 排放变量，维度为 `Time, emissions_zdim, south_north, west_east`。
- WRF 网格、投影、模拟开始时间、排放来源和单位说明等全局属性。

气体和 VOC 变量单位为：

```text
mol km^-2 hr^-1
```

气溶胶变量单位为：

```text
ug m^-2 s^-1
```

## 运行检查建议

运行前建议检查：

1. `echo $WRFHOME` 或 PowerShell 中 `$env:WRFHOME` 是否正确。
2. `$WRFHOME/WRF/run/` 是否存在 `wrfinput_d*`。
3. `$WRFHOME/peizhishuju/MEIC2023/MEICMM/` 是否存在目标月份的物种文件。
4. 当前运行目录是否有 `meictowrf.xlsx`。
5. `meictowrf.xlsx` 的时间分配列长度是否至少覆盖 24 小时。
6. `PM10` 是否大于或等于 `PM25`，否则粗模态排放会被截断为 0。

可用以下方式快速检查输出文件内容：

```bash
ncdump -h $WRFHOME/WRF/run/wrfchemi_00z_d01
```

也可以用 Python 检查变量：

```python
import netCDF4 as nc

path = "wrfchemi_00z_d01"
with nc.Dataset(path) as ds:
    print(ds.dimensions.keys())
    print(ds.variables.keys())
    print(ds.variables["E_CO"].shape)
```

## 已知注意事项

- `merged.py` 当前不能直接运行，需要先修正缩进错误和缺失的 `sys` 导入。
- `merged.py` 和 `meictowrf.py` 的默认 MEIC 目录不完全一致，运行前需要统一。
- `meictowrf.py` 接收“预报时长”参数，但当前没有使用该参数控制输出时长。
- `meictowrf.py` 固定生成 `00z` 和 `12z` 两个 12 小时排放文件。
- 气体、VOC 和气溶胶的单位转换依赖输入文件的实际单位，运行前应确认 MEIC 文件中的单位与脚本假设一致。
- 若某些物种文件缺失，脚本会给出警告并用零场或估算方式继续处理，最终输出文件仍可能生成，但对应物种可能为 0 或估算值。
