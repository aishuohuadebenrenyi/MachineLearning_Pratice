# encoding: UTF-8

import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve
import warnings
# 忽略警告信息
warnings.filterwarnings('ignore')

# 加载数据
data = pd.read_csv('creditcard.csv')

# 数据探索性分析及可视化
print(data.describe())
# 画图
plt.rcParams['font.sans-serif']=['Hiragino Sans GB'] # 设置字体，正常显示中文标签
plt.figure(figsize=(8, 6))
sns.countplot(x = 'Class', data = data)
plt.title('类别分类 \n(0: 正常; 1: 欺诈)')
plt.show()
# 显示交易笔数、欺诈交易笔数、欺诈交易比例
all_num = len(data)
fraud_num = len(data[data['Class'] == 1])
print('总交易数: ', all_num)
print('欺诈交易数: ', fraud_num)
print('欺诈交易数占总交易数的占比: {:.6f}'.format(fraud_num/all_num))
# 直方图可视化欺诈和正常交易
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))
bins = 50
ax1.hist(data.Time[data.Class == 1], bins=bins, color='deeppink')
ax1.set_title('诈骗交易')
ax2.hist(data.Time[data.Class == 0], bins=bins, color='deepskyblue')
ax2.set_title('正常交易')
plt.xlabel('时间')
plt.ylabel('交易次数')
plt.show()


# 数据预处理
# 对 Amount 进行数据规范化
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
# 特征选择
target = np.array(data.Class.tolist())
feathers = data.drop(['Time', 'Class'], axis=1).values
# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(feathers, target, test_size=0.1, random_state=33)


# 逻辑回归模型训练
lg = LogisticRegression()
lg.fit(train_x, train_y)
predict_y = lg.predict(test_x)


# 模型评估
# 预测样本的置信分数
score_y = lg.decision_function(test_x)
# 计算混淆矩阵并显示
cm = confusion_matrix(test_y, predict_y)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix') 
plt.colorbar()
trick_marks = [0, 1]
plt.xticks(trick_marks, rotation = 0)
plt.yticks(trick_marks)
thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 
    plt.text(j, i, cm[i, j], 
        horizontalalignment = 'center', 
        color = 'white' if cm[i, j] > thresh else 'black') 
plt.tight_layout() 
plt.ylabel('True label') 
plt.xlabel('Predicted label') 
plt.show()

# 输出精确率、召回率、F1值
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]
TN = cm[0,0]
precision = TP/(TP + FP)
recall = TP/(TP + FN)
print('准确度: {:.3f}\n召回率: {:.3f}\nF1值: {:.3f}'.format(precision, recall, 2*precision*recall/(precision + recall)))

# 绘制 精确率-召回率 曲线
precisions, recalls, thresholds = precision_recall_curve(test_y, score_y)
plt.step(recalls, precisions, color = 'b', alpha = 0.2, where = 'post') 
plt.fill_between(recalls, precisions, step ='post', alpha = 0.2, color = 'b') 
plt.plot(recalls, precisions, linewidth=2) 
plt.xlim([0.0,1]) 
plt.ylim([0.0,1.05]) 
plt.xlabel('召回率') 
plt.ylabel('精确率') 
plt.title('精确率-召回率 曲线') 
plt.show()








