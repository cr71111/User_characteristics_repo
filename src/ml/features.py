# -*- coding: utf-8 -*-
"""
===============================================================================
ML特征提取器 v2.9 — 从lifecycle层提取ML-ready特征向量
===============================================================================
职责：将lifecycle层宽表转换为机器学习可用的特征矩阵。

设计原则：
  1. 自动列名匹配 — 兼容不同版本的lifecycle输出字段名
  2. 缺失值处理 — 中位数填充 + 标记缺失
  3. 特征工程 — 自动构造比值/衍生特征
  4. 标准化输出 — 统一返回 (特征矩阵, 特征名列表, 元数据)

v2.9 更新:
  - 新增行为特征组（时间熵值、路线曲折系数、速度变异系数等）
  - 新增功率特征（峰值功率）
  - 适配车辆形态三级分级和众包/专送增强判定

使用方式：
    from ml_features import FeatureExtractor
    extractor = FeatureExtractor()
    X, feature_names, meta = extractor.fit_transform(df_lifecycle)
===============================================================================
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


FEATURE_GROUPS = {
    '骑行强度': {
        'desc': '反映用户骑行活跃程度',
        'features': [
            ('近7d单合约日均骑行时长_h', ['近7d单合约日均骑行时长_h', '近7d日均骑行时长_h']),
            ('近7d单合约日均行驶里程_km', ['近7d单合约日均行驶里程_km', '近7d日均行驶里程_km']),
            ('近7d日均骑行次数', ['近7d日均骑行次数']),
            ('近7d日均行驶距离_km', ['近7d日均行驶距离_km']),
            ('近7d出勤天数', ['近7d出勤天数']),
            ('近7d出勤率', ['近7d出勤率']),
        ]
    },
    '速度特征': {
        'desc': '反映骑行速度水平和极端值',
        'features': [
            ('近7d平均骑行速度_kmh', ['近7d平均骑行速度_kmh']),
            ('近7d_P50骑行速度_kmh', ['近7d_P50骑行速度_kmh']),
            ('近7d_P90骑行速度_kmh', ['近7d_P90骑行速度_kmh']),
            ('近7d最高速度_kmh', ['近7d最高速度_kmh']),
            ('近7d夜间骑行均速_kmh', ['近7d夜间骑行均速_kmh']),
        ]
    },
    '电流特征': {
        'desc': '反映电流使用强度和异常程度',
        'features': [
            ('近7d平均骑行电流_A', ['近7d平均骑行电流_A']),
            ('近7d最大电流_A', ['近7d最大电流_A']),
            ('近7d日均电流超60A次数', ['近7d日均电流超60A次数']),
            ('近7d日均电流超80A次数', ['近7d日均电流超80A次数']),
            ('近7d日均超100A连续次数', ['近7d日均超100A连续次数']),
            ('近7d日均超100A累计时长_h', ['近7d日均超100A累计时长_h']),
        ]
    },
    '电耗特征': {
        'desc': '反映用电效率和消耗水平',
        'features': [
            ('近7d单合约日均用电量_kWh', ['近7d单合约日均用电量_kWh']),
            ('近7d百公里电耗_kWh', ['近7d百公里电耗_kWh']),
            ('近7d单合约日均放电时长_h', ['近7d单合约日均放电时长_h']),
            ('近7d单合约日均怠速放电_h', ['近7d单合约日均怠速放电_h']),
        ]
    },
    'SOC特征': {
        'desc': '反映电池使用习惯和深度放电程度',
        'features': [
            ('近7d平均骑行SOC', ['近7d平均骑行SOC']),
            ('近7d最低SOC', ['近7d最低SOC']),
            ('近7d取电时平均SOC', ['近7d取电时平均SOC']),
            ('近7d还电时平均SOC', ['近7d还电时平均SOC']),
            ('近7d单次换电SOC消耗', ['近7d单次换电SOC消耗']),
            ('近7d_SOC低于20%时长占比', ['近7d_SOC低于20%时长占比']),
            ('近7d_SOC低于10%时长占比', ['近7d_SOC低于10%时长占比']),
        ]
    },
    '换电特征': {
        'desc': '反映换电频率和时段偏好',
        'features': [
            ('近7d日均换电次数', ['近7d日均换电次数']),
            ('近7d平峰换电占比', ['近7d平峰换电占比']),
            ('近7d深夜换电占比', ['近7d深夜换电占比']),
        ]
    },
    '时段分布': {
        'desc': '反映骑行时段偏好',
        'features': [
            ('近7d高峰骑行占比', ['近7d高峰骑行占比']),
            ('近7d日均午间高峰长时骑行次数', ['近7d日均午间高峰长时骑行次数']),
            ('近7d日均晚间高峰长时骑行次数', ['近7d日均晚间高峰长时骑行次数']),
            ('近7d日均平峰长时骑行次数', ['近7d日均平峰长时骑行次数']),
            ('近7d日均夜间长时骑行次数', ['近7d日均夜间长时骑行次数']),
        ]
    },
    '温度特征': {
        'desc': '反映电池工作温度',
        'features': [
            ('近7d最高温度_℃', ['近7d最高温度_℃']),
        ]
    },
    '活动范围': {
        'desc': '反映用户活动半径和空间分布',
        'features': [
            ('近7d最大活动半径_km', ['近7d最大活动半径_km']),
            ('近7d_R90活动半径_km', ['近7d_R90活动半径_km']),
            ('近7d凸包覆盖面积_km2', ['近7d凸包覆盖面积_km2']),
            ('近7d最大单次出行距离_km', ['近7d最大单次出行距离_km']),
        ]
    },
    '行为规律': {
        'desc': '反映用户骑行行为的规律性和模式',
        'features': [
            ('近7d平均上线时间熵值', ['近7d平均上线时间熵值']),
            ('近7d平均骑行时段集中度', ['近7d平均骑行时段集中度']),
            ('近7d平均路线曲折系数', ['近7d平均路线曲折系数']),
            ('近7d平均速度变异系数', ['近7d平均速度变异系数']),
            ('近7d平均静止时长占比', ['近7d平均静止时长占比']),
            ('近7d最大跨区域转移次数', ['近7d最大跨区域转移次数']),
        ]
    },
    '功率特征': {
        'desc': '反映电池输出功率和改装风险',
        'features': [
            ('近7d峰值功率_W', ['近7d峰值功率_W']),
        ]
    },
}

DERIVED_FEATURES = [
    ('速度波动比', '近7d_P90骑行速度_kmh', '近7d_P50骑行速度_kmh',
     'P90/P50速度比，>2.0说明速度波动大，疑似改装'),
    ('电流波动比', '近7d最大电流_A', '近7d平均骑行电流_A',
     '最大/平均电流比，>3.0说明电流波动大'),
    ('怠速放电占比', '近7d单合约日均怠速放电_h', '近7d单合约日均放电时长_h',
     '怠速放电/总放电，>0.5疑似地摊/储能'),
    ('深夜骑行占比', '近7d日均夜间长时骑行次数', '近7d日均骑行次数',
     '夜间骑行次数占比，>0.5为夜猫子型'),
    ('换电频率强度', '近7d日均换电次数', '近7d单合约日均用电量_kWh',
     '换电次数/用电量，>2说明频繁换电但用电少，疑似电池老化'),
    ('SOC消耗率', '近7d单次换电SOC消耗', '近7d单合约日均骑行时长_h',
     'SOC消耗/骑行时长，>20说明耗电异常快'),
    ('高速骑行密度', '近7d日均高速骑行点数', '近7d日均骑行次数',
     '高速点数/骑行次数，>0.3说明频繁高速'),
    ('功率电流比', '近7d峰值功率_W', '近7d平均骑行电流_A',
     '峰值功率/平均电流，>200疑似高压改装车（电动摩托车或改装车）'),
    ('骑行规律性', '近7d出勤率', '近7d平均上线时间熵值',
     '出勤率/时间熵值，值越高越规律，专送骑手通常更高'),
    ('空间集中度', '近7d_R90活动半径_km', '近7d凸包覆盖面积_km2',
     'R90/凸包面积，值越低越集中，专送骑手活动范围小且固定'),
]


class FeatureExtractor:
    """从lifecycle层DataFrame提取ML特征矩阵"""

    def __init__(self, fillna_method: str = 'median'):
        self.fillna_method = fillna_method
        self.feature_names_: List[str] = []
        self.derived_names_: List[str] = []
        self.fill_values_: Dict[str, float] = {}
        self.meta_: Dict = {}

    def _resolve_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """从候选列名中找到实际存在的列"""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict]:
        """
        从lifecycle DataFrame提取特征矩阵

        参数:
            df: lifecycle层输出DataFrame

        返回:
            X: 特征矩阵 (numpy array)
            feature_names: 所有特征名列表
            meta: 元数据 (特征组信息、缺失统计等)
        """
        df = df.copy()
        n_samples = len(df)

        feature_data = {}
        missing_report = {}
        all_feature_names = []

        for group_name, group_info in FEATURE_GROUPS.items():
            for feat_name, candidates in group_info['features']:
                col = self._resolve_column(df, candidates)
                if col is not None:
                    series = pd.to_numeric(df[col], errors='coerce')
                    missing_count = series.isna().sum()
                    if missing_count > 0:
                        missing_report[feat_name] = {
                            'missing_count': int(missing_count),
                            'missing_pct': round(missing_count / n_samples * 100, 1),
                            'group': group_name,
                        }
                    feature_data[feat_name] = series.values
                    all_feature_names.append(feat_name)

        self.feature_names_ = all_feature_names

        X = np.column_stack([feature_data[name] for name in all_feature_names])

        if self.fillna_method == 'median':
            for j, name in enumerate(all_feature_names):
                col_data = X[:, j]
                mask = np.isnan(col_data) | np.isinf(col_data)
                if mask.any():
                    median_val = np.nanmedian(col_data)
                    col_data[mask] = median_val
                    X[:, j] = col_data
                    self.fill_values_[name] = float(median_val)

        X_derived, derived_names = self._compute_derived_features(df, X, all_feature_names)
        self.derived_names_ = derived_names

        X_full = np.column_stack([X, X_derived]) if X_derived.shape[1] > 0 else X
        all_names = all_feature_names + derived_names

        self.meta_ = {
            'n_samples': n_samples,
            'n_features_raw': len(all_feature_names),
            'n_features_derived': len(derived_names),
            'n_features_total': len(all_names),
            'feature_groups': {
                g: [f[0] for f in info['features'] if f[0] in all_feature_names]
                for g, info in FEATURE_GROUPS.items()
            },
            'derived_features': [d[0] for d in DERIVED_FEATURES if d[0] in derived_names],
            'missing_report': missing_report,
            'fill_values': self.fill_values_,
        }

        return X_full, all_names, self.meta_

    def _compute_derived_features(
        self, df: pd.DataFrame, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """计算衍生特征（比值类）"""
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        derived_list = []
        derived_names = []

        for feat_name, num_col, den_col, _ in DERIVED_FEATURES:
            num_candidates = self._get_candidates(num_col)
            den_candidates = self._get_candidates(den_col)

            num_idx = None
            den_idx = None
            for c in num_candidates:
                if c in name_to_idx:
                    num_idx = name_to_idx[c]
                    break
            for c in den_candidates:
                if c in name_to_idx:
                    den_idx = name_to_idx[c]
                    break

            if num_idx is not None and den_idx is not None:
                num_vals = X[:, num_idx]
                den_vals = X[:, den_idx]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(den_vals > 0, num_vals / den_vals, 0)
                ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
                derived_list.append(ratio)
                derived_names.append(feat_name)

        if derived_list:
            return np.column_stack(derived_list), derived_names
        return np.zeros((X.shape[0], 0)), []

    def _get_candidates(self, col_name: str) -> List[str]:
        """获取某特征的所有候选列名"""
        for group_info in FEATURE_GROUPS.values():
            for feat_name, candidates in group_info['features']:
                if feat_name == col_name:
                    return candidates
        return [col_name]

    def get_feature_importance_hints(self) -> Dict[str, str]:
        """返回特征重要性的人工先验提示"""
        hints = {}
        for group_name, group_info in FEATURE_GROUPS.items():
            for feat_name, _ in group_info['features']:
                if feat_name in self.feature_names_:
                    hints[feat_name] = group_info['desc']

        for feat_name, _, _, desc in DERIVED_FEATURES:
            if feat_name in self.derived_names_:
                hints[feat_name] = desc

        return hints


def extract_features_from_parquet(parquet_path: str) -> Tuple[np.ndarray, List[str], Dict, pd.DataFrame]:
    """
    便捷函数：从parquet文件直接提取特征

    返回:
        X, feature_names, meta, df_original
    """
    df = pd.read_parquet(parquet_path)
    extractor = FeatureExtractor()
    X, feature_names, meta = extractor.fit_transform(df)
    return X, feature_names, meta, df