# -*- coding: utf-8 -*-
"""
===============================================================================
ML统一运行入口 v1.0 — 支持按步骤选择执行
===============================================================================

执行模式:
  anomaly   — 无监督异常检测 (孤立森林 + DBSCAN)
  label     — 标注管理 (生成待标注 / 保存标注 / 查看统计)
  train     — XGBoost 监督分类器训练
  predict   — 对新数据预测异常类型
  full      — 一键全流程: anomaly → label generate (人工标注后继续 train → predict)
  pipeline  — 完整闭环: anomaly → label generate → (等待人工) → train → predict

使用示例:
  python src/ml/run_ml.py anomaly
  python src/ml/run_ml.py anomaly --contamination 0.03
  python src/ml/run_ml.py label generate
  python src/ml/run_ml.py label save --input <path>
  python src/ml/run_ml.py label stats
  python src/ml/run_ml.py train
  python src/ml/run_ml.py predict --model <path>
  python src/ml/run_ml.py full
===============================================================================
"""

import os
import sys
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import EXPORT_PATH_LIFECYCLE_7D, DATA_OUTPUT_ROOT

LIFECYCLE_PATH = EXPORT_PATH_LIFECYCLE_7D
LABELED_DATA_PATH = os.path.join(DATA_OUTPUT_ROOT, 'ml_anomaly', 'labels', 'labeled_data.csv')


def _get_latest_anomaly_report():
    report_dir = os.path.join(DATA_OUTPUT_ROOT, 'ml_anomaly')
    if not os.path.isdir(report_dir):
        return None
    reports = [f for f in os.listdir(report_dir)
               if f.startswith('anomaly_report_') and f.endswith('.csv')
               and '_cross_validation' not in f and '_new_findings' not in f]
    if not reports:
        return None
    reports.sort(reverse=True)
    return os.path.join(report_dir, reports[0])


def _get_latest_model():
    model_dir = os.path.join(DATA_OUTPUT_ROOT, 'ml_anomaly', 'models')
    if not os.path.isdir(model_dir):
        return None
    models = [f for f in os.listdir(model_dir)
              if f.startswith('xgb_model_') and f.endswith('.json')
              and '_meta' not in f and '_encoder' not in f]
    if not models:
        return None
    models.sort(reverse=True)
    return os.path.join(model_dir, models[0])


def _get_latest_to_label():
    label_dir = os.path.join(DATA_OUTPUT_ROOT, 'ml_anomaly', 'labels')
    if not os.path.isdir(label_dir):
        return None
    sheets = [f for f in os.listdir(label_dir)
              if f.startswith('to_label_') and f.endswith('.csv')]
    if not sheets:
        return None
    sheets.sort(reverse=True)
    return os.path.join(label_dir, sheets[0])


def run_anomaly(lifecycle_path=None, contamination=0.05):
    print("\n" + "=" * 60)
    print("  [ML-1] 无监督异常检测")
    print("=" * 60)

    lifecycle_path = lifecycle_path or LIFECYCLE_PATH

    if not os.path.exists(lifecycle_path):
        print(f"❌ lifecycle 数据不存在: {lifecycle_path}")
        print(f"   请先运行: python src/pipeline/run_pipeline.py --mode full")
        return None

    from src.ml.anomaly import AnomalyDetector, run_anomaly_detection
    import pandas as pd

    print(f"📥 输入: {lifecycle_path}")
    df = pd.read_parquet(lifecycle_path)
    print(f"   用户数: {len(df)}")

    result_df, detector = run_anomaly_detection(df, contamination=contamination, save=True)
    return result_df


def run_label_generate(report_path=None, n_top=50):
    print("\n" + "=" * 60)
    print("  [ML-2a] 生成待标注文件")
    print("=" * 60)

    report_path = report_path or _get_latest_anomaly_report()

    if not report_path or not os.path.exists(report_path):
        print("❌ 未找到异常检测报告，请先运行 anomaly 步骤")
        return None

    from src.ml.labeler import LabelManager

    print(f"📥 输入: {report_path}")
    manager = LabelManager()
    manager.generate_labeling_sheet(
        anomaly_report_path=report_path,
        n_top=n_top,
        only_new_findings=True,
    )

    to_label = _get_latest_to_label()
    if to_label:
        print(f"\n📝 请打开以下文件，填写「人工标注」列:")
        print(f"   {to_label}")
        print(f"\n   可选值: 正常 | 改装/超速 | 地摊/储能 | 电池老化 | 暴力驾驶 | 其他异常 | 不确定")

    return to_label


def run_label_save(input_path=None):
    print("\n" + "=" * 60)
    print("  [ML-2b] 保存标注结果")
    print("=" * 60)

    input_path = input_path or _get_latest_to_label()

    if not input_path or not os.path.exists(input_path):
        print("❌ 未找到待标注文件，请先运行 label generate 步骤")
        return

    from src.ml.labeler import LabelManager

    print(f"📥 输入: {input_path}")
    manager = LabelManager()
    manager.save_labels(labeled_csv_path=input_path)


def run_label_stats():
    print("\n" + "=" * 60)
    print("  [ML-2c] 标注统计")
    print("=" * 60)

    from src.ml.labeler import LabelManager

    manager = LabelManager()
    manager.print_stats()


def run_train(labels_path=None, lifecycle_path=None, test_size=0.2):
    print("\n" + "=" * 60)
    print("  [ML-3] XGBoost 监督分类器训练")
    print("=" * 60)

    labels_path = labels_path or LABELED_DATA_PATH
    lifecycle_path = lifecycle_path or LIFECYCLE_PATH

    if not os.path.exists(labels_path):
        print(f"❌ 标注数据不存在: {labels_path}")
        print(f"   需要至少 50 条标注才能有效训练")
        print(f"   请先完成人工标注并执行: python src/ml/run_ml.py label save")
        return None

    if not os.path.exists(lifecycle_path):
        print(f"❌ lifecycle 数据不存在: {lifecycle_path}")
        return None

    from src.ml.classifier import SupervisedClassifier
    import pandas as pd

    print(f"📥 标注数据: {labels_path}")
    print(f"📥 lifecycle: {lifecycle_path}")

    df_labels = pd.read_csv(labels_path, encoding='utf-8-sig')
    df_lifecycle = pd.read_parquet(lifecycle_path)

    labeled_count = len(df_labels)
    print(f"   标注数: {labeled_count}")

    if labeled_count < 10:
        print(f"⚠️  标注数量 ({labeled_count}) 过少，模型效果可能极差，建议 ≥ 50 条")

    classifier = SupervisedClassifier()
    metrics = classifier.train(df_labels, df_lifecycle, test_size=test_size)
    classifier.print_train_results()

    model_path = classifier.save_model()

    print(f"\n📌 下一步: python src/ml/run_ml.py predict --model \"{model_path}\"")
    return model_path


def run_predict(lifecycle_path=None, model_path=None, confidence=0.6):
    print("\n" + "=" * 60)
    print("  [ML-4] XGBoost 预测")
    print("=" * 60)

    lifecycle_path = lifecycle_path or LIFECYCLE_PATH
    model_path = model_path or _get_latest_model()

    if not model_path or not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        print(f"   请先运行: python src/ml/run_ml.py train")
        return None

    if not os.path.exists(lifecycle_path):
        print(f"❌ lifecycle 数据不存在: {lifecycle_path}")
        return None

    from src.ml.classifier import SupervisedClassifier
    import pandas as pd

    print(f"📥 lifecycle: {lifecycle_path}")
    print(f"📥 模型: {model_path}")

    classifier = SupervisedClassifier()
    classifier.load_model(model_path)

    df_lifecycle = pd.read_parquet(lifecycle_path)
    result = classifier.predict(df_lifecycle, confidence_threshold=confidence)

    low_conf = classifier.get_low_confidence_samples(result)
    print(f"\n⚠️  低置信度样本 (需人工审核): {len(low_conf)} 人")
    if len(low_conf) > 0:
        print(low_conf[['用户id', 'ml_xgb_预测类别', 'ml_xgb_置信度']].head(10).to_string(index=False))

    pred_dir = os.path.join(DATA_OUTPUT_ROOT, 'ml_anomaly', 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(pred_dir, f'prediction_{timestamp}.csv')
    result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n📁 预测结果: {output_path}")

    return result


def run_full(contamination=0.05, n_top=50):
    """一键全流程: anomaly → generate labels → (等待人工) → 提示下一步"""
    print("\n" + "=" * 60)
    print("  ML 全流程 (半自动)")
    print("=" * 60)

    # Step 1+2: 异常检测 + 生成待标注
    run_anomaly(contamination=contamination)
    to_label = run_label_generate(n_top=n_top)

    print("\n" + "-" * 60)
    print("  ⏸️  请完成人工标注后继续")
    print("-" * 60)
    print(f"""
  1. 打开文件: {to_label}
  2. 在「人工标注」列填写标注结果
  3. 保存文件
  4. 运行以下命令继续:

     python src/ml/run_ml.py label save
     python src/ml/run_ml.py train
     python src/ml/run_ml.py predict
""")


def main():
    parser = argparse.ArgumentParser(
        description='ML 统一运行入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/ml/run_ml.py anomaly                        # 异常检测
  python src/ml/run_ml.py anomaly --contamination 0.03   # 调整异常比例
  python src/ml/run_ml.py anomaly --lifecycle <自定义路径>
  python src/ml/run_ml.py label generate                  # 生成待标注
  python src/ml/run_ml.py label generate --n-top 100      # 取Top100
  python src/ml/run_ml.py label save                      # 保存标注(自动找最新)
  python src/ml/run_ml.py label save --input <路径>       # 保存指定文件
  python src/ml/run_ml.py label stats                     # 查看标注统计
  python src/ml/run_ml.py train                           # 训练模型
  python src/ml/run_ml.py predict                         # 预测(自动找最新模型)
  python src/ml/run_ml.py predict --model <模型路径>       # 指定模型
  python src/ml/run_ml.py full                            # 一键: 检测+生成
"""
    )

    subparsers = parser.add_subparsers(dest='mode', help='执行模式')

    # ---- anomaly ----
    anomaly_parser = subparsers.add_parser('anomaly', help='无监督异常检测')
    anomaly_parser.add_argument('--lifecycle', type=str, default=None,
                                help=f'lifecycle路径 (默认: {LIFECYCLE_PATH})')
    anomaly_parser.add_argument('--contamination', type=float, default=0.05,
                                help='预期异常比例 (默认: 0.05)')

    # ---- label ----
    label_parser = subparsers.add_parser('label', help='标注管理')
    label_sub = label_parser.add_subparsers(dest='label_action', help='标注子命令')

    label_gen = label_sub.add_parser('generate', help='生成待标注文件')
    label_gen.add_argument('--report', type=str, default=None,
                           help='异常报告路径 (默认: 自动查找最新)')
    label_gen.add_argument('--n-top', type=int, default=50,
                           help='取Top-N最异常用户 (默认: 50)')

    label_save = label_sub.add_parser('save', help='保存标注结果')
    label_save.add_argument('--input', type=str, default=None,
                            help='标注CSV路径 (默认: 自动查找最新)')

    label_sub.add_parser('stats', help='查看标注统计')

    # ---- train ----
    train_parser = subparsers.add_parser('train', help='训练XGBoost分类器')
    train_parser.add_argument('--labels', type=str, default=None,
                              help=f'标注数据路径 (默认: {LABELED_DATA_PATH})')
    train_parser.add_argument('--lifecycle', type=str, default=None,
                              help=f'lifecycle路径 (默认: {LIFECYCLE_PATH})')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                              help='测试集比例 (默认: 0.2)')

    # ---- predict ----
    predict_parser = subparsers.add_parser('predict', help='预测新数据')
    predict_parser.add_argument('--lifecycle', type=str, default=None,
                                help=f'lifecycle路径 (默认: {LIFECYCLE_PATH})')
    predict_parser.add_argument('--model', type=str, default=None,
                                help='模型路径 (默认: 自动查找最新)')
    predict_parser.add_argument('--confidence', type=float, default=0.6,
                                help='低置信度阈值 (默认: 0.6)')

    # ---- full ----
    full_parser = subparsers.add_parser('full', help='一键全流程 (anomaly + label generate)')
    full_parser.add_argument('--contamination', type=float, default=0.05,
                             help='预期异常比例 (默认: 0.05)')
    full_parser.add_argument('--n-top', type=int, default=50,
                             help='取Top-N (默认: 50)')

    args = parser.parse_args()

    if args.mode == 'anomaly':
        run_anomaly(
            lifecycle_path=args.lifecycle,
            contamination=args.contamination,
        )

    elif args.mode == 'label':
        if args.label_action == 'generate':
            run_label_generate(
                report_path=args.report,
                n_top=args.n_top,
            )
        elif args.label_action == 'save':
            run_label_save(input_path=args.input)
        elif args.label_action == 'stats':
            run_label_stats()
        else:
            label_parser.print_help()

    elif args.mode == 'train':
        run_train(
            labels_path=args.labels,
            lifecycle_path=args.lifecycle,
            test_size=args.test_size,
        )

    elif args.mode == 'predict':
        run_predict(
            lifecycle_path=args.lifecycle,
            model_path=args.model,
            confidence=args.confidence,
        )

    elif args.mode == 'full':
        run_full(
            contamination=args.contamination,
            n_top=args.n_top,
        )

    else:
        parser.print_help()
        print("\n❌ 请指定执行模式，例如: python src/ml/run_ml.py anomaly")


if __name__ == '__main__':
    main()