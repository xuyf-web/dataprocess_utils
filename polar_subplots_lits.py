import matplotlib.pyplot as plt 
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
import numpy as np
import pandas as pd
import cmaps   # 🔑 NCL colormap
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import make_interp_spline
from matplotlib import rcParams
import geopandas as gpd
from rasterio.features import rasterize

rcParams['font.family'] = 'Arial'

# ================== 读取 TIFF 文件 ==================
tif_file = "merged_LST_Summer_010_95.tif"
with rasterio.open(tif_file) as src:
    band_data = src.read(1).astype(float)
    band_data[band_data == src.nodata] = np.nan
    lon_min, lat_min, lon_max, lat_max = src.bounds
    lon_plt = np.linspace(lon_min, lon_max, src.width)
    lat_plt = np.linspace(lat_max, lat_min, src.height)

lon_plt, lat_plt = np.meshgrid(lon_plt, lat_plt)

# ================== 读取 Shapefile 并创建掩膜 ==================
shp_paths = [
    "Boreal Forests.shp",
    "Tundra.shp"
]

# 读取所有生态区 shapefile
gdf_all = []
for shp in shp_paths:
    gdf_all.append(gpd.read_file(shp))
gdf_all = gpd.GeoDataFrame(pd.concat(gdf_all, ignore_index=True), crs=gdf_all[0].crs)

# 栅格化掩膜
eco_mask = rasterize(
    [(geom, 1) for geom in gdf_all.geometry],
    out_shape=(band_data.shape[0], band_data.shape[1]),  # 输出掩膜的尺寸与原数据一致
    transform=src.transform,
    fill=0,
    dtype='uint8'
).astype(bool)

# 应用掩膜，只保留两个区域内的数据
band_data[~eco_mask] = np.nan

# ================== 绘图主图 ==================
fig = plt.figure(figsize=(4.5, 4.5))
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())

# ================== 圆形裁剪 ==================
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# ================== 底图 ==================
land = cfeature.NaturalEarthFeature(
    'physical', 'land', '110m',
    edgecolor='none',
    facecolor='#e2e2e2',  # 浅灰色
)
ax.add_feature(land, zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='black', zorder=5)

# ================== 绘制 TIFF 数据 ==================
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
mesh = ax.pcolormesh(
    lon_plt, lat_plt, band_data,
    cmap=cmaps.BlueYellowRed,
    norm=norm,
    transform=ccrs.PlateCarree(), zorder=3
)

# ================== 颜色条 ==================
cax = fig.add_axes([0.25, 0.01, 0.5, 0.03])
cbar = plt.colorbar(mesh, cax=cax, orientation="horizontal", extend='both')
cbar.set_label(r"$\Delta$LST (K)", fontsize=10)
cbar.set_ticks(np.arange(-1, 1.01, 0.5))

# ✅ 格式化刻度标签：0 显示为 '0'，其余保留两位小数
tick_labels = [f"{t:.1f}" if t != 0 else "0" for t in cbar.get_ticks()]
cbar.set_ticklabels(tick_labels)

cbar.ax.tick_params(labelsize=10)
cbar.outline.set_linewidth(0.5)

# ================== 纬度同心圆 ==================
for lat in [40, 50, 60, 70, 80]:
    lons = np.linspace(-180, 180, 361)
    lats = np.full_like(lons, lat)
    ax.plot(lons, lats, transform=ccrs.PlateCarree(),
            color='gray', linewidth=0.4, linestyle='--', alpha=0.6, zorder=4)
    ax.text(180, lat, f'{lat}°N', transform=ccrs.PlateCarree(),
            ha='center', va='bottom', fontsize=11, color='black')

# ================== 经度放射线 + 经度标注 ==================
skip_lons = [0, -180, -60]
for lon in np.arange(-180, 180, 60):
    ax.plot([lon, lon], [40, 90], transform=ccrs.PlateCarree(),
            color='gray', linewidth=0.4, linestyle='--', alpha=0.6, zorder=4)
    if lon in skip_lons:
        continue
    label = f"{abs(lon)}°W" if lon < 0 else f"{lon}°E"
    ax.text(lon, 35, label, transform=ccrs.PlateCarree(),
            ha='center', va='bottom', fontsize=11, color='black')

# ================== 小柱状图子图 ==================
inset_ax = fig.add_axes([0.135, 0.105, 0.31, 0.185])  # left, bottom, width, height

# ================== 准备柱状图数据 ==================
pos_color = "#c05444"
neg_color = "#6291ab"
edge_linewidth = 0.4

with rasterio.open(tif_file) as src:
    data = src.read(1).astype(float)
    data[data == src.nodata] = np.nan
    height, width = data.shape

# ================== 应用掩膜并获取有效数据 ==================
# 将掩膜展平为一维数组
eco_mask_flattened = eco_mask.flatten()

# 获取数据并应用掩膜
data_flattened = data.flatten()  # 将二维数据展平为一维
data_masked = data_flattened[eco_mask_flattened]  # 只保留掩膜区域内的数据

# 去除 NaN 值
data_masked = data_masked[~np.isnan(data_masked)]

# ================== 分箱 ==================
bins = np.array([-np.inf, -1, -0.8, -0.6, -0.4, -0.2, 0,
                 0.2, 0.4, 0.6, 0.8, 1.0, np.inf])
counts = [np.sum((data_masked > bins[i]) & (data_masked <= bins[i+1])) for i in range(len(bins)-1)]
counts = np.array(counts)
mid_points = (bins[:-1] + bins[1:]) / 2

# 排除 inf 的 mid_points 用于绘图和颜色映射
finite_mask = np.isfinite(mid_points)
finite_mid_points = mid_points[finite_mask]
finite_counts = counts[finite_mask]

# ================== 颜色映射 ==================
cmap = cmaps.cmp_b2r
colors = [cmap((mp - (-1)) / 2) for mp in finite_mid_points]

# 绘制柱状图
for i in range(len(counts)):
    left = bins[i] if np.isfinite(bins[i]) else -1.2
    right = bins[i+1] if np.isfinite(bins[i+1]) else 1.2
    color = cmap((mid_points[i] - (-1)) / 2) if np.isfinite(mid_points[i]) else 'grey'
    inset_ax.bar(
        left,
        counts[i],
        width=(right-left),
        color=color,
        edgecolor='black',
        linewidth=edge_linewidth,
        align='edge'
    )

# 平滑曲线
if len(finite_mid_points) > 3:
    x_new = np.linspace(finite_mid_points.min(), finite_mid_points.max(), 200)
    spl = make_interp_spline(finite_mid_points, finite_counts, k=3)
    y_smooth = spl(x_new)
    inset_ax.plot(x_new, y_smooth, color='black', linewidth=1)


# 计算占比
pos_count = counts[mid_points > 0].sum()
neg_count = counts[mid_points <= 0].sum()
total_count = pos_count + neg_count
pos_pct = pos_count / total_count * 100
neg_pct = neg_count / total_count * 100

label_fontsize = 9
inset_ax.text(-1.25, max(counts)*1.9, f"{neg_pct:.1f}% ≤ 0", color=neg_color,
              fontsize=label_fontsize, ha='left', va='top')
inset_ax.text(1.25, max(counts)*1.9, f"{pos_pct:.1f}% > 0", color=pos_color,
              fontsize=label_fontsize, ha='right', va='top')


# ================== 坐标轴样式 ==================
inset_ax.set_xticks([-1, 0, 1])
inset_ax.set_xticklabels(['-1', '0', '1'], fontsize=9)
inset_ax.set_yticks(np.arange(0, 801, 200))
inset_ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8'], fontsize=9)
inset_ax.set_ylabel("Pixel Count (×10³)", fontsize=9)
inset_ax.axvline(x=0, color='red', linewidth=1.5, linestyle='--')
inset_ax.set_ylim(0, 800)


plt.savefig("LST-JJA.png", bbox_inches='tight', dpi=1000)
plt.show()