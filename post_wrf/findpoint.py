import numpy as np

def _distance_squared(stn_lon, stn_lat, lon2d, lat2d):
    difflon = stn_lon - lon2d
    difflat = stn_lat - lat2d
    return np.square(difflon) + np.square(difflat)

def nearest_position(stn_lon, stn_lat, lon2d, lat2d):
    """
    获取最临近格点坐标索引
    """
    
    distances_squared = _distance_squared(stn_lon, stn_lat, lon2d, lat2d)
    
    # 获取最小距离的索引
    min_index = np.unravel_index(np.argmin(distances_squared), distances_squared.shape)
    
    return min_index

def nearest_positions(stn_lon, stn_lat, lon2d, lat2d, num=4):
    """
    获取最临近的指定数量num个格点坐标索引
    """
    
    distances_squared = _distance_squared(stn_lon, stn_lat, lon2d, lat2d)
    flat_distances = distances_squared.ravel()
    total_points = flat_distances.size
    num = max(1, min(int(num), total_points))

    # 使用argpartition优化大网格下的性能，再对候选索引做局部排序以保持距离升序
    candidate_indices = np.argpartition(flat_distances, num - 1)[:num]
    local_sort = np.argsort(flat_distances[candidate_indices])
    flat_indices = candidate_indices[local_sort]
    indices = np.unravel_index(flat_indices, distances_squared.shape)
    
    return indices

def weighted_average(stn_lon, stn_lat, lon2d, lat2d, data2d, num=4):
    """
    计算近邻格点的加权平均值
    """
    
    # 获取最临近的num个格点坐标索引
    indices = nearest_positions(stn_lon, stn_lat, lon2d, lat2d, num=num)
    
    # 提取num个格点的数据
    flat_indices = np.ravel_multi_index(indices, data2d.shape)
    points_data = data2d.flat[flat_indices]

    # 计算num个格点到站点的距离
    distances_squared = _distance_squared(
        stn_lon,
        stn_lat,
        lon2d.flat[flat_indices],
        lat2d.flat[flat_indices],
    )

    # 过滤无效值，避免单个NaN传播为整体NaN
    valid_mask = np.isfinite(points_data) & np.isfinite(distances_squared)
    if not np.any(valid_mask):
        return np.nan

    points_data = points_data[valid_mask]
    distances_squared = distances_squared[valid_mask]

    # 若站点与格点重合，直接返回重合点均值
    zero_dist_mask = distances_squared == 0
    if np.any(zero_dist_mask):
        return np.mean(points_data[zero_dist_mask])

    # 计算权重（距离倒数作为权重）
    weights = 1 / distances_squared
    weights /= np.sum(weights)

    # 加权平均
    weighted_avg = np.sum(points_data * weights)

    return weighted_avg

if __name__ == '__main__':
    # 生成测试数据
    lon2d, lat2d = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    data2d = np.random.rand(11, 11)
    stn_lon = 3.3
    stn_lat = 7.8
    
    # 提取最临近格点坐标索引
    min_index = nearest_position(stn_lon, stn_lat, lon2d, lat2d)
    print('nearest index = ',min_index)
    
    # 提取最临近的指定数量num个格点坐标索引
    indices = nearest_positions(stn_lon, stn_lat, lon2d, lat2d, num=4)
    print('nearest indices = ',indices)
    
    # 计算加权平均
    weighted_avg = weighted_average(stn_lon, stn_lat, lon2d, lat2d, data2d, num=4)
    print(weighted_avg)
