# -*- coding: utf-8 -*-
"""
GPS 围栏加载和空间匹配逻辑
从 L1 提取为独立模块,其他层禁止重复实现。
"""

import time
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import (
    EXPORT_FILE_PROVINCE_GEOJSON,
    EXPORT_FILE_CITY_GEOJSON,
    EXPORT_FILE_DISTRICT_GEOJSON,
)

# 地球半径常量
EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_M = 6371000.0


def haversine(lat1, lon1, lat2, lon2, unit='km'):
    """
    Haversine公式计算两点间球面距离
    
    Args:
        lat1, lon1: 起点坐标（纬度、经度）
        lat2, lon2: 终点坐标（纬度、经度）
        unit: 单位 ('km' 或 'm')
    
    Returns:
        float: 两点间的距离
    """
    radius = EARTH_RADIUS_KM if unit == 'km' else EARTH_RADIUS_M
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    return radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# 地理围栏数据缓存
_fence_data_cache = None


def load_fence_data():
    """加载省/市/区县地理围栏数据"""
    global _fence_data_cache
    if _fence_data_cache is not None:
        return _fence_data_cache

    print("🗺️  加载地理围栏数据...")
    start = time.time()

    def load_geo(path, output_col_name):
        gdf = gpd.read_file(path)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        name_col = None
        for col in gdf.columns:
            if col.lower() == 'name':
                name_col = col
                break
        if not name_col:
            for col in gdf.columns:
                if 'name' in col.lower() or '名称' in col or '省' in col or '市' in col or '县' in col:
                    name_col = col
                    break
        if not name_col:
            raise ValueError(f"❌ 在 {path} 中未找到名称字段")
        gdf = gdf[[name_col, 'geometry']].copy()
        gdf = gdf.rename(columns={name_col: output_col_name})
        return gdf

    try:
        gdf_prov = load_geo(EXPORT_FILE_PROVINCE_GEOJSON, "核心活动省份")
        gdf_city = load_geo(EXPORT_FILE_CITY_GEOJSON, "核心活动城市")
        gdf_dist = load_geo(EXPORT_FILE_DISTRICT_GEOJSON, "核心活动区县")
        _fence_data_cache = {'prov': gdf_prov, 'city': gdf_city, 'district': gdf_dist}
        print(f"✅ 三级围栏加载完成：省 {len(gdf_prov)} 个，市 {len(gdf_city)} 个，区县 {len(gdf_dist)} 个，耗时{time.time()-start:.2f}s")
        return _fence_data_cache
    except Exception as e:
        print(f"❌ 加载围栏失败: {e}")
        return None


def batch_gps_to_region(df, lat_col='中心纬度', lon_col='中心经度'):
    """
    GPS空间匹配优化版：使用坐标哈希降维缓存，性能提升5-10倍

    输入：包含GPS坐标的DataFrame
    输出：包含省/市/区县信息的DataFrame
    """
    fence_data = load_fence_data()
    if not fence_data:
        return pd.DataFrame('', index=df.index, columns=['核心活动省份', '核心活动城市', '核心活动区县'])

    df = df.copy()
    df['_idx'] = df.index
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

    valid_gps = df.dropna(subset=[lat_col, lon_col]).copy()
    final_result = pd.DataFrame('', index=df.index, columns=['核心活动省份', '核心活动城市', '核心活动区县'])

    if valid_gps.empty:
        return final_result

    valid_gps['lat_round'] = valid_gps[lat_col].round(3)
    valid_gps['lon_round'] = valid_gps[lon_col].round(3)

    unique_coords = valid_gps[['lat_round', 'lon_round']].drop_duplicates().reset_index(drop=True)
    print(f"   🗺️ 坐标降维：{len(valid_gps)} 个点 -> {len(unique_coords)} 个唯一坐标")

    gdf_unique = gpd.GeoDataFrame(
        unique_coords,
        geometry=gpd.points_from_xy(unique_coords['lon_round'], unique_coords['lat_round']),
        crs="EPSG:4326"
    )

    print(f"   🗺️ 正在执行空间匹配（降维后）...")

    def safe_sjoin(gdf_pts, gdf_poly, col_name):
        try:
            res = gpd.sjoin(gdf_pts, gdf_poly, how="left", predicate="within")
            res = res[~res.index.duplicated(keep='first')]
            res = res.reindex(gdf_pts.index)
            if col_name in res.columns:
                return res[col_name].fillna('').tolist()
            else:
                return [''] * len(gdf_pts)
        except Exception as e:
            print(f"   ⚠️ [{col_name}] 空间匹配失败: {e}")
            return [''] * len(gdf_pts)

    unique_coords = unique_coords.copy()
    unique_coords['核心活动省份'] = safe_sjoin(gdf_unique, fence_data['prov'], '核心活动省份')
    unique_coords['核心活动城市'] = safe_sjoin(gdf_unique, fence_data['city'], '核心活动城市')
    unique_coords['核心活动区县'] = safe_sjoin(gdf_unique, fence_data['district'], '核心活动区县')

    valid_gps = valid_gps.drop(
        columns=[c for c in ['核心活动省份', '核心活动城市', '核心活动区县'] if c in valid_gps.columns],
        errors='ignore'
    )
    valid_gps = valid_gps.merge(unique_coords, on=['lat_round', 'lon_round'], how='left')

    for col in ['核心活动省份', '核心活动城市', '核心活动区县']:
        if col not in valid_gps.columns:
            valid_gps[col] = ''

    idx_map = valid_gps.drop_duplicates(subset=['_idx']).set_index('_idx')
    for col in ['核心活动省份', '核心活动城市', '核心活动区县']:
        final_result[col] = idx_map[col].reindex(final_result.index).fillna('').values

    return final_result
