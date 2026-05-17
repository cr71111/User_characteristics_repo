# -*- coding: utf-8 -*-
"""
===============================================================================
ML异常检测器 v1.0 — 孤立森林 + DBSCAN 双模型异常发现
===============================================================================
职责：在规则判定之外，用无监督学习发现"规则抓不到的异常"。

模型：
  1. IsolationForest — 孤立森林异常检测
     原理：异常点特征值极端，用很少的分割就能隔离出来
     输出：异常分数（越低越异常）+ 二分类标签

  2. DBSCAN — 基于密度的用户分群
     原理：不预设分类，让数据自己聚类，发现新用户群体
     输出：聚类标签（-1=噪声点/异常）

使用方式：
    from ml_anomaly import AnomalyDetector
    detector = AnomalyDetector(contamination=0.05)
    result_df = detector.fit_predict(df_lifecycle)

    # 查看双模型交叉验证结果
    cross = detector.get_cross_validation()
    # 查看新发现的异常用户（模型判异常但规则判正常）
    new_findings = detector.get_new_findings(df_lifecycle)
===============================================================================
"""

import os
import sys
import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import DATA_OUTPUT_ROOT

ANOMALY_OUTPUT_DIR = os.path.join(DATA_OUTPUT_ROOT, 'ml_anomaly')

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn 未安装，请运行: pip install scikit-learn")


class AnomalyDetector:
    """
    ML异常检测器

    参数:
        contamination: 预期异常比例 (默认0.05 = 5%)
        dbscan_eps: DBSCAN邻域半径 (默认0.5)
        dbscan_min_samples: DBSCAN最小样本数 (默认10)
        random_state: 随机种子
    """

    def __init__(
        self,
        contamination: float = 0.05,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 10,
        random_state: int = 42,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn 未安装，请运行: pip install scikit-learn")

        self.contamination = contamination
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.random_state = random_state

        self.scaler: Optional[StandardScaler] = None
        self.if_model: Optional[IsolationForest] = None
        self.dbscan_model: Optional[DBSCAN] = None
        self.pca: Optional[PCA] = None

        self.feature_names_: List[str] = []
        self.X_scaled_: Optional[np.ndarray] = None
        self.result_df_: Optional[pd.DataFrame] = None

    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        对lifecycle数据执行异常检测

        参数:
            df: lifecycle层DataFrame（需包含用户id列）
            feature_cols: 指定特征列名，None则自动选择数值列

        返回:
            包含异常检测结果的DataFrame（新增列）
        """
        from src.ml_features import FeatureExtractor

        df = df.copy()

        extractor = FeatureExtractor()
        X, feature_names, meta = extractor.fit_transform(df)
        self.feature_names_ = feature_names

        self.scaler = StandardScaler()
        self.X_scaled_ = self.scaler.fit_transform(X)

        self.if_model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
            max_samples='auto',
        )
        if_labels = self.if_model.fit_predict(self.X_scaled_)
        if_scores = self.if_model.score_samples(self.X_scaled_)

        df['ml_异常标签_iforest'] = if_labels
        df['ml_异常分数_iforest'] = np.round(if_scores, 4)
        df['ml_异常程度'] = np.where(
            if_labels == -1,
            np.where(if_scores < np.percentile(if_scores, 25), '高', '中'),
            '正常'
        )

        self.dbscan_model = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
        )
        db_labels = self.dbscan_model.fit_predict(self.X_scaled_)
        df['ml_聚类标签_dbscan'] = db_labels
        df['ml_是否噪声点'] = (db_labels == -1).astype(int)

        n_components = min(10, X.shape[1])
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(self.X_scaled_)
        for i in range(n_components):
            df[f'ml_PCA_{i+1}'] = np.round(X_pca[:, i], 4)

        df['ml_双模型异常'] = (
            (df['ml_异常标签_iforest'] == -1) & (df['ml_是否噪声点'] == 1)
        ).astype(int)

        self.result_df_ = df

        return df

    def get_cross_validation(self) -> pd.DataFrame:
        """双模型交叉验证：对比孤立森林 vs DBSCAN"""
        if self.result_df_ is None:
            raise ValueError("请先调用 fit_predict()")

        df = self.result_df_
        total = len(df)

        both_anomaly = (df['ml_异常标签_iforest'] == -1) & (df['ml_是否噪声点'] == 1)
        if_only = (df['ml_异常标签_iforest'] == -1) & (df['ml_是否噪声点'] == 0)
        dbscan_only = (df['ml_异常标签_iforest'] == 1) & (df['ml_是否噪声点'] == 1)
        both_normal = (df['ml_异常标签_iforest'] == 1) & (df['ml_是否噪声点'] == 0)

        cross = pd.DataFrame([
            {'交叉类型': '双模型一致异常', '用户数': both_anomaly.sum(),
             '占比%': round(both_anomaly.sum() / total * 100, 2),
             '说明': '两个模型都认为是异常，置信度最高'},
            {'交叉类型': '仅孤立森林异常', '用户数': if_only.sum(),
             '占比%': round(if_only.sum() / total * 100, 2),
             '说明': '孤立森林判异常但DBSCAN未识别，可能是边界异常'},
            {'交叉类型': '仅DBSCAN噪声点', '用户数': dbscan_only.sum(),
             '占比%': round(dbscan_only.sum() / total * 100, 2),
             '说明': 'DBSCAN判为噪声但孤立森林未识别，可能是小众群体'},
            {'交叉类型': '双模型一致正常', '用户数': both_normal.sum(),
             '占比%': round(both_normal.sum() / total * 100, 2),
             '说明': '两个模型都认为正常'},
        ])

        return cross

    def get_new_findings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        发现规则漏判的异常用户

        返回：模型判异常但规则判正常的用户列表
        """
        if self.result_df_ is None:
            raise ValueError("请先调用 fit_predict()")

        result = self.result_df_.copy()

        rule_anomaly_cols = []
        for col in ['风险标签', '用户等级_动态', '用户形态']:
            if col in result.columns:
                rule_anomaly_cols.append(col)

        if rule_anomaly_cols:
            is_rule_normal = pd.Series(True, index=result.index)
            if '风险标签' in result.columns:
                is_rule_normal &= (result['风险标签'] == '正常')
            if '用户等级_动态' in result.columns:
                is_rule_normal &= ~result['用户等级_动态'].isin(['高损耗用户', '暴力'])
            if '用户形态' in result.columns:
                is_rule_normal &= ~result['用户形态'].isin(['改装/超速车', '地摊/储能'])

            new_findings = result[
                (result['ml_异常标签_iforest'] == -1) & is_rule_normal
            ].copy()
        else:
            new_findings = result[result['ml_异常标签_iforest'] == -1].copy()

        sort_col = 'ml_异常分数_iforest'
        new_findings = new_findings.sort_values(sort_col)

        return new_findings

    def get_cluster_profiles(self) -> pd.DataFrame:
        """DBSCAN聚类画像：每个簇的特征均值"""
        if self.result_df_ is None:
            raise ValueError("请先调用 fit_predict()")

        df = self.result_df_
        labels = df['ml_聚类标签_dbscan']

        profiles = []
        for label in sorted(labels.unique()):
            cluster_df = df[labels == label]
            profile = {'聚类标签': int(label), '用户数': len(cluster_df)}

            for feat in self.feature_names_:
                if feat in cluster_df.columns:
                    profile[f'{feat}_均值'] = round(cluster_df[feat].mean(), 2)

            profiles.append(profile)

        return pd.DataFrame(profiles)

    def get_top_anomalies(
        self, n: int = 20, min_score: Optional[float] = None
    ) -> pd.DataFrame:
        """获取Top-N最异常用户"""
        if self.result_df_ is None:
            raise ValueError("请先调用 fit_predict()")

        df = self.result_df_[self.result_df_['ml_异常标签_iforest'] == -1].copy()
        df = df.sort_values('ml_异常分数_iforest')

        if min_score is not None:
            df = df[df['ml_异常分数_iforest'] <= min_score]

        return df.head(n)

    def explain_anomaly(self, user_id) -> Dict:
        """解释某个用户为什么被判定为异常（基于特征偏离度）"""
        if self.result_df_ is None or self.X_scaled_ is None:
            raise ValueError("请先调用 fit_predict()")

        df = self.result_df_
        user_row = df[df['用户id'] == user_id]
        if len(user_row) == 0:
            return {'error': f'用户 {user_id} 不存在'}

        idx = user_row.index[0]
        user_vec = self.X_scaled_[idx]

        global_mean = self.X_scaled_.mean(axis=0)
        global_std = self.X_scaled_.std(axis=0)
        global_std = np.where(global_std == 0, 1, global_std)

        deviations = (user_vec - global_mean) / global_std

        top_deviations = []
        for i, dev in enumerate(deviations):
            if abs(dev) > 1.5:
                direction = '偏高' if dev > 0 else '偏低'
                top_deviations.append({
                    '特征': self.feature_names_[i],
                    '偏离度': round(float(dev), 2),
                    '方向': direction,
                    '用户值': round(float(df.iloc[idx].get(self.feature_names_[i], 0)), 2),
                    '全局均值': round(float(global_mean[i] * (global_std[i] if global_std[i] != 1 else 1) + global_mean[i]), 2),
                })

        top_deviations.sort(key=lambda x: abs(x['偏离度']), reverse=True)

        return {
            '用户id': user_id,
            '异常分数': float(user_row['ml_异常分数_iforest'].values[0]),
            '异常程度': user_row['ml_异常程度'].values[0],
            '是否双模型异常': bool(user_row['ml_双模型异常'].values[0]),
            'Top偏离特征': top_deviations[:5],
        }

    def save_report(self, output_path: Optional[str] = None) -> str:
        """保存异常检测报告"""
        if self.result_df_ is None:
            raise ValueError("请先调用 fit_predict()")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(ANOMALY_OUTPUT_DIR, exist_ok=True)

        if output_path is None:
            output_path = os.path.join(ANOMALY_OUTPUT_DIR, f'anomaly_report_{timestamp}.csv')

        report_cols = ['用户id', 'ml_异常标签_iforest', 'ml_异常分数_iforest',
                       'ml_异常程度', 'ml_聚类标签_dbscan', 'ml_是否噪声点',
                       'ml_双模型异常']

        extra_cols = []
        for col in ['风险标签', '用户等级_动态', '用户形态', '重评分数',
                     '用户生命周期状态_7d', '近7d出勤天数', '近7d出勤率']:
            if col in self.result_df_.columns:
                extra_cols.append(col)

        save_cols = report_cols + extra_cols
        save_cols = [c for c in save_cols if c in self.result_df_.columns]

        self.result_df_[save_cols].to_csv(output_path, index=False, encoding='utf-8-sig')

        cross = self.get_cross_validation()
        cross_path = output_path.replace('.csv', '_cross_validation.csv')
        cross.to_csv(cross_path, index=False, encoding='utf-8-sig')

        new_findings = self.get_new_findings(self.result_df_)
        findings_path = output_path.replace('.csv', '_new_findings.csv')
        new_findings[save_cols].to_csv(findings_path, index=False, encoding='utf-8-sig')

        return output_path


def run_anomaly_detection(
    df: pd.DataFrame,
    contamination: float = 0.05,
    save: bool = True,
) -> Tuple[pd.DataFrame, AnomalyDetector]:
    """
    便捷函数：一键运行异常检测

    返回:
        (result_df, detector)
    """
    detector = AnomalyDetector(contamination=contamination)
    result_df = detector.fit_predict(df)

    print(f"\n{'='*60}")
    print("📊 ML异常检测结果")
    print(f"{'='*60}")

    cross = detector.get_cross_validation()
    print("\n双模型交叉验证:")
    for _, row in cross.iterrows():
        print(f"  {row['交叉类型']}: {row['用户数']}人 ({row['占比%']}%)")

    new_findings = detector.get_new_findings(df)
    print(f"\n🔍 新发现（模型判异常但规则判正常）: {len(new_findings)} 人")

    top = detector.get_top_anomalies(10)
    print(f"\n⚠️  Top-10 最异常用户:")
    for _, row in top.iterrows():
        uid = row.get('用户id', 'N/A')
        score = row.get('ml_异常分数_iforest', 'N/A')
        level = row.get('用户等级_动态', 'N/A') if '用户等级_动态' in row.index else 'N/A'
        print(f"  用户{uid}: 分数={score}, 等级={level}")

    if save:
        path = detector.save_report()
        print(f"\n📁 报告已保存: {path}")

    return result_df, detector


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ML异常检测')
    parser.add_argument('--input', type=str, required=True,
                        help='lifecycle层parquet文件路径')
    parser.add_argument('--contamination', type=float, default=0.05,
                        help='预期异常比例 (默认: 0.05)')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存报告文件')
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    run_anomaly_detection(df, contamination=args.contamination, save=not args.no_save)