# -*- coding: utf-8 -*-
"""01-基础配套实验①：机器学习评估口径的真实数字（三大范式与评估篇附表）。
零下载：只用 sklearn 内置数据；随机种子固定 random_state=0，任何机器结果一致。
运行：python code/scripts/ml_eval_demo.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score,
                             precision_recall_fscore_support, confusion_matrix,
                             mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score)
from sklearn.datasets import make_classification

rng = 0  # 固定种子：全文所有数字可复现

print("=" * 62)
print("实验A · 类别不平衡陷阱：accuracy 会骗你")
print("=" * 62)
# 造 5% 正类的分类问题（1000 样本：950 负 / 50 正）
X, y = make_classification(n_samples=1000, n_features=12, n_informative=8,
                           n_redundant=3, weights=[0.95, 0.05], random_state=rng)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                      stratify=y, random_state=rng)
clf = LogisticRegression(max_iter=2000, random_state=rng).fit(Xtr, ytr)
pred = clf.predict(Xte)
baseline = np.zeros_like(yte)          # "无脑全猜负类"的对照
acc_clf = accuracy_score(yte, pred)
acc_base = accuracy_score(yte, baseline)
print(f"训练集分布: 负类 {int((ytr == 0).sum())}  正类 {int((ytr == 1).sum())}")
print(f"[分类器] accuracy = {acc_clf:.4f}   [全猜负类] accuracy = {acc_base:.4f}")
print("正类(少数)实际抓到的样本数:",
      int(((yte == 1) & (pred == 1)).sum()), f"/ {int((yte == 1).sum())}")
print(f"→ 只看 accuracy 显得 '{acc_clf:.1%}' 很漂亮，但真正的正类几乎全漏掉。")
print(f"宏平均 F1 = {precision_recall_fscore_support(yte, pred, average='macro')[2]:.4f}")
print(f"ROC-AUC  = {roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]):.4f}")
print(f"(AUC 与 threshold 无关，能看出'排序上'正类是否高；这也是它比 accuracy 稳的原因)")
print()

print("=" * 62)
print("实验B · 正确评估口径：同一个模型，四套指标说话")
print("=" * 62)
print("classification_report（测试集 200 样本）：")
print(classification_report(yte, pred, digits=3))
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
print(f"混淆矩阵 [[TN {tn}, FP {fp}], [FN {fn}, TP {tp}]]")
cv_scores = cross_val_score(
    LogisticRegression(max_iter=2000, random_state=rng), X, y, cv=5,
    scoring='roc_auc')
print(f"5 折交叉验证 ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print()

print("=" * 62)
print("实验C · 回归口径：R² 与『预测均值』这个下限")
print("=" * 62)
Xb, yb = load_diabetes(return_X_y=True)   # 442 样本·10 特征，内置
Xtrb, Xteb, ytrb, yteb = train_test_split(Xb, yb, test_size=0.2, random_state=rng)
lr = LinearRegression().fit(Xtrb, ytrb)
predb = lr.predict(Xteb)
const = np.full_like(yteb, yteb.mean())   # 基线：永远输出训练目标均值
print(f"[线性回归]  MSE = {mean_squared_error(yteb, predb):.2f}  MAE = {mean_absolute_error(yteb, predb):.2f}  R² = {r2_score(yteb, predb):.4f}")
print(f"[预测均值]  R²  = {r2_score(yteb, const):.4f}   <- 规则：任何模型先战胜'均值下限'才有意义")
print()

print("=" * 62)
print("实验D · 无监督没有 ground truth：轮廓系数只是'像不像一坨'")
print("=" * 62)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
Xd, _ = load_breast_cancer(return_X_y=True)
Xd = Xd[:, :6]  # 取 6 个数值维，演示足够
sil = {}
for k in (2, 3, 5):
    km = KMeans(n_clusters=k, n_init=10, random_state=rng).fit(Xd)
    sil[k] = silhouette_score(Xd, km.labels_)
print("KMeans 轮廓系数:", {k: round(v, 4) for k, v in sil.items()})
print("（轮廓系数衡量'类内紧、类间远'，它不回答'分几类才是对的'——那要看业务）")
print()
print("done · 全部数字可在本机一键复现（python code/scripts/ml_eval_demo.py）")
