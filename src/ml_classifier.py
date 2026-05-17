# -*- coding: utf-8 -*-
"""
===============================================================================
ML监督分类器 v1.0 — XGBoost多分类 + 特征重要性 + 模型持久化
===============================================================================
职责：基于人工标注数据训练监督学习模型，实现异常类型自动分类。

功能：
  1. 加载标注数据 + 特征提取 → 训练集
  2. XGBoost多分类训练（改装/地摊/电池老化/暴力/正常）
  3. 特征重要性排名 + SHAP解释
  4. 模型保存/加载
  5. 新数据预测 + 低置信度标记（反馈入队）
  6. 与孤立森林结果对比

使用方式：
    # 训练模型
    python src/ml_classifier.py train --labels ./data/output/ml_anomaly/labels/labeled_data.csv
           --lifecycle ./data/output/lifecycle/user_7d.parquet

    # 预测新数据
    python src/ml_classifier.py predict --lifecycle ./data/output/lifecycle/user_7d.parquet
           --model ./data/output/ml_anomaly/models/xgb_model_xxx.json

    # 查看特征重要性
    python src/ml_classifier.py importance --model ./data/output/ml_anomaly/models/xgb_model_xxx.json
===============================================================================
"""

import os
import sys
import json
import pickle
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

MODEL_OUTPUT_DIR = os.path.join(DATA_OUTPUT_ROOT, 'ml_anomaly', 'models')

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import LabelEncoder
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  xgboost 未安装，请运行: pip install xgboost")


LABEL_TO_ID = {
    '正常': 0,
    '改装/超速': 1,
    '地摊/储能': 2,
    '电池老化': 3,
    '暴力驾驶': 4,
    '其他异常': 5,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

ANOMALY_LABELS = ['改装/超速', '地摊/储能', '电池老化', '暴力驾驶', '其他异常']


class SupervisedClassifier:
    """
    XGBoost监督分类器

    参数:
        n_estimators: 树的数量
        max_depth: 树的最大深度
        learning_rate: 学习率
        random_state: 随机种子
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        if not XGB_AVAILABLE:
            raise ImportError("xgboost 未安装，请运行: pip install xgboost")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model: Optional[xgb.XGBClassifier] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names_: List[str] = []
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.train_metrics_: Dict = {}
        self.n_classes_: int = 0

    def train(
        self,
        df_labeled: pd.DataFrame,
        df_lifecycle: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Dict:
        """
        训练XGBoost分类器

        参数:
            df_labeled: 标注数据 (需包含 用户id, 人工标注 列)
            df_lifecycle: lifecycle层全量数据 (用于提取特征)
            test_size: 测试集比例

        返回:
            训练指标字典
        """
        from src.ml_features import FeatureExtractor

        if '用户id' not in df_labeled.columns or '人工标注' not in df_labeled.columns:
            raise ValueError("标注数据必须包含 '用户id' 和 '人工标注' 列")

        if '用户id' not in df_lifecycle.columns:
            raise ValueError("lifecycle数据必须包含 '用户id' 列")

        df_labeled['用户id'] = df_labeled['用户id'].astype(str)
        df_lifecycle['用户id'] = df_lifecycle['用户id'].astype(str)

        df_labeled = df_labeled[df_labeled['人工标注'].isin(LABEL_TO_ID.keys())].copy()

        labeled_ids = set(df_labeled['用户id'].tolist())
        df_users = df_lifecycle[df_lifecycle['用户id'].isin(labeled_ids)].copy()

        if len(df_users) == 0:
            raise ValueError("标注用户与lifecycle数据无交集，请检查用户id是否匹配")

        extractor = FeatureExtractor()
        X, feature_names, meta = extractor.fit_transform(df_users)
        self.feature_names_ = feature_names

        self.label_encoder = LabelEncoder()
        y_labels = df_users['用户id'].map(
            dict(zip(df_labeled['用户id'], df_labeled['人工标注']))
        )
        y = self.label_encoder.fit_transform(y_labels)
        self.n_classes_ = len(self.label_encoder.classes_)

        class_counts = pd.Series(y).value_counts().sort_index()
        min_class_count = class_counts.min()

        if min_class_count < 2:
            stratify = None
            print(f"⚠️  最少类别样本数={min_class_count}，无法分层抽样，使用随机划分")
        else:
            stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify,
        )

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0,
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )

        try:
            cv = StratifiedKFold(n_splits=min(5, min_class_count), shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except Exception:
            cv_mean = accuracy
            cv_std = 0.0

        importance_df = pd.DataFrame({
            '特征': self.feature_names_,
            '重要性': np.round(self.model.feature_importances_, 4),
        }).sort_values('重要性', ascending=False)
        self.feature_importance_ = importance_df

        self.train_metrics_ = {
            'n_samples': len(y),
            'n_features': len(self.feature_names_),
            'n_classes': self.n_classes_,
            'classes': self.label_encoder.classes_.tolist(),
            'class_distribution': {
                self.label_encoder.classes_[i]: int(class_counts.get(i, 0))
                for i in range(self.n_classes_)
            },
            'accuracy': round(accuracy, 4),
            'cv_accuracy_mean': round(cv_mean, 4),
            'cv_accuracy_std': round(cv_std, 4),
            'classification_report': report,
            'top_features': importance_df.head(10).to_dict('records'),
        }

        return self.train_metrics_

    def predict(
        self,
        df_lifecycle: pd.DataFrame,
        confidence_threshold: float = 0.6,
    ) -> pd.DataFrame:
        """
        对新数据进行预测

        参数:
            df_lifecycle: lifecycle层数据
            confidence_threshold: 低置信度阈值（低于此值标记为待审核）

        返回:
            包含预测结果的DataFrame
        """
        if self.model is None:
            raise ValueError("模型未训练或未加载，请先调用 train() 或 load_model()")

        from src.ml_features import FeatureExtractor

        df = df_lifecycle.copy()

        extractor = FeatureExtractor()
        X, feature_names, meta = extractor.fit_transform(df)

        missing_features = set(self.feature_names_) - set(feature_names)
        if missing_features:
            print(f"⚠️  训练时有但当前数据缺失的特征: {missing_features}")

        common_features = [f for f in self.feature_names_ if f in feature_names]
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        X_aligned = np.column_stack([X[:, name_to_idx[f]] for f in common_features])

        y_pred = self.model.predict(X_aligned)
        y_proba = self.model.predict_proba(X_aligned)

        df['ml_xgb_预测类别'] = self.label_encoder.inverse_transform(y_pred)
        df['ml_xgb_置信度'] = np.round(np.max(y_proba, axis=1), 4)
        df['ml_xgb_低置信度'] = (df['ml_xgb_置信度'] < confidence_threshold).astype(int)

        for i, cls_name in enumerate(self.label_encoder.classes_):
            df[f'ml_xgb_概率_{cls_name}'] = np.round(y_proba[:, i], 4)

        return df

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """获取特征重要性排名"""
        if self.feature_importance_ is None:
            raise ValueError("模型未训练，请先调用 train()")
        return self.feature_importance_.head(top_n)

    def get_low_confidence_samples(
        self, df_predicted: pd.DataFrame
    ) -> pd.DataFrame:
        """
        获取低置信度样本（需要人工审核）

        返回:
            低置信度用户列表
        """
        if 'ml_xgb_低置信度' not in df_predicted.columns:
            raise ValueError("请先调用 predict()")

        low_conf = df_predicted[df_predicted['ml_xgb_低置信度'] == 1].copy()

        cols = ['用户id', 'ml_xgb_预测类别', 'ml_xgb_置信度']
        prob_cols = [c for c in df_predicted.columns if c.startswith('ml_xgb_概率_')]
        cols += prob_cols

        extra_cols = ['风险标签', '用户等级_动态', '用户形态']
        for c in extra_cols:
            if c in low_conf.columns:
                cols.append(c)

        cols = [c for c in cols if c in low_conf.columns]
        return low_conf[cols].sort_values('ml_xgb_置信度')

    def compare_with_iforest(self, df_predicted: pd.DataFrame) -> pd.DataFrame:
        """
        对比XGBoost预测 vs 孤立森林结果

        返回:
            对比统计DataFrame
        """
        if 'ml_异常标签_iforest' not in df_predicted.columns:
            raise ValueError("数据中缺少孤立森林结果列，请先运行异常检测")

        df = df_predicted.copy()

        is_if_anomaly = df['ml_异常标签_iforest'] == -1
        is_xgb_anomaly = df['ml_xgb_预测类别'].isin(ANOMALY_LABELS)

        total = len(df)

        both_anomaly = (is_if_anomaly & is_xgb_anomaly).sum()
        if_only = (is_if_anomaly & ~is_xgb_anomaly).sum()
        xgb_only = (~is_if_anomaly & is_xgb_anomaly).sum()
        both_normal = (~is_if_anomaly & ~is_xgb_anomaly).sum()

        comparison = pd.DataFrame([
            {'对比类型': '双模型一致异常', '用户数': int(both_anomaly),
             '占比%': round(both_anomaly / total * 100, 2),
             '说明': '孤立森林+XGBoost都判异常，置信度最高'},
            {'对比类型': '仅孤立森林异常', '用户数': int(if_only),
             '占比%': round(if_only / total * 100, 2),
             '说明': '孤立森林判异常但XGBoost判正常，可能是孤立森林误判'},
            {'对比类型': '仅XGBoost异常', '用户数': int(xgb_only),
             '占比%': round(xgb_only / total * 100, 2),
             '说明': 'XGBoost判异常但孤立森林未识别，监督模型的新发现'},
            {'对比类型': '双模型一致正常', '用户数': int(both_normal),
             '占比%': round(both_normal / total * 100, 2),
             '说明': '两个模型都判正常'},
        ])

        return comparison

    def save_model(self, output_path: Optional[str] = None) -> str:
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train()")

        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if output_path is None:
            output_path = os.path.join(MODEL_OUTPUT_DIR, f'xgb_model_{timestamp}.json')

        self.model.save_model(output_path)

        meta = {
            'timestamp': timestamp,
            'feature_names': self.feature_names_,
            'label_classes': self.label_encoder.classes_.tolist() if self.label_encoder else [],
            'train_metrics': self.train_metrics_,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
            },
        }
        meta_path = output_path.replace('.json', '_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        encoder_path = output_path.replace('.json', '_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print(f"模型已保存: {output_path}")
        print(f"元数据: {meta_path}")
        print(f"标签编码器: {encoder_path}")

        return output_path

    def load_model(self, model_path: str) -> Dict:
        """
        加载已训练的模型

        参数:
            model_path: 模型JSON文件路径

        返回:
            模型元数据
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        meta_path = model_path.replace('.json', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.feature_names_ = meta.get('feature_names', [])
            self.train_metrics_ = meta.get('train_metrics', {})
        else:
            meta = {}

        encoder_path = model_path.replace('.json', '_encoder.pkl')
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.n_classes_ = len(self.label_encoder.classes_)

        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            if self.feature_names_:
                self.feature_importance_ = pd.DataFrame({
                    '特征': self.feature_names_,
                    '重要性': np.round(self.model.feature_importances_, 4),
                }).sort_values('重要性', ascending=False)

        return meta

    def print_train_results(self):
        """打印训练结果"""
        if not self.train_metrics_:
            print("暂无训练结果。")
            return

        m = self.train_metrics_

        print(f"\n{'='*60}")
        print("XGBoost 训练结果")
        print(f"{'='*60}")
        print(f"样本数: {m['n_samples']}")
        print(f"特征数: {m['n_features']}")
        print(f"类别数: {m['n_classes']}")
        print(f"\n类别分布:")
        for cls_name, count in m['class_distribution'].items():
            print(f"  [{cls_name}]: {count} 条")
        print(f"\n准确率: {m['accuracy']*100:.1f}%")
        print(f"交叉验证准确率: {m['cv_accuracy_mean']*100:.1f}% (±{m['cv_accuracy_std']*100:.1f}%)")

        print(f"\nTop-10 重要特征:")
        for i, feat in enumerate(m.get('top_features', [])[:10]):
            print(f"  {i+1}. {feat['特征']}: {feat['重要性']:.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ML监督分类器')
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--labels', type=str, required=True,
                              help='标注数据CSV路径')
    train_parser.add_argument('--lifecycle', type=str, required=True,
                              help='lifecycle层parquet路径')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                              help='测试集比例 (默认: 0.2)')
    train_parser.add_argument('--output', type=str, default=None,
                              help='模型保存路径')

    predict_parser = subparsers.add_parser('predict', help='预测新数据')
    predict_parser.add_argument('--lifecycle', type=str, required=True,
                                help='lifecycle层parquet路径')
    predict_parser.add_argument('--model', type=str, required=True,
                                help='模型文件路径')
    predict_parser.add_argument('--confidence', type=float, default=0.6,
                                help='低置信度阈值 (默认: 0.6)')
    predict_parser.add_argument('--output', type=str, default=None,
                                help='预测结果保存路径')

    importance_parser = subparsers.add_parser('importance', help='查看特征重要性')
    importance_parser.add_argument('--model', type=str, required=True,
                                   help='模型文件路径')
    importance_parser.add_argument('--top-n', type=int, default=20,
                                   help='显示Top-N特征 (默认: 20)')

    args = parser.parse_args()

    classifier = SupervisedClassifier()

    if args.command == 'train':
        df_labels = pd.read_csv(args.labels, encoding='utf-8-sig')
        df_lifecycle = pd.read_parquet(args.lifecycle)

        metrics = classifier.train(df_labels, df_lifecycle, test_size=args.test_size)
        classifier.print_train_results()

        model_path = classifier.save_model(output_path=args.output)
        print(f"\n下一步: python src/ml_classifier.py predict --lifecycle <路径> --model {model_path}")

    elif args.command == 'predict':
        classifier.load_model(args.model)
        df_lifecycle = pd.read_parquet(args.lifecycle)

        result = classifier.predict(df_lifecycle, confidence_threshold=args.confidence)

        low_conf = classifier.get_low_confidence_samples(result)
        print(f"\n低置信度样本 (需人工审核): {len(low_conf)} 人")
        if len(low_conf) > 0:
            print(low_conf.head(10).to_string(index=False))

        if args.output:
            result.to_csv(args.output, index=False, encoding='utf-8-sig')
            print(f"\n预测结果已保存: {args.output}")

    elif args.command == 'importance':
        classifier.load_model(args.model)
        imp = classifier.get_feature_importance(top_n=args.top_n)
        print(f"\nTop-{args.top_n} 特征重要性:")
        for i, (_, row) in enumerate(imp.iterrows()):
            bar = '█' * int(row['重要性'] * 100)
            print(f"  {i+1:>2}. {row['特征']:<30} {row['重要性']:.4f} {bar}")

    else:
        parser.print_help()