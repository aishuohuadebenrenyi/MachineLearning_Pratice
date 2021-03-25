# encoding: UTF-8

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns

# 数据加载
data = pd.read_csv('credit_card_info.csv')

# 数据探索
print(data.describe()) # 数据集概览
# 查看下一个月违约率的情况
next_moth_payment = data['default.payment.next.month'].value_counts()
# 画图可视化
df = pd.DataFrame({'default.payment.next.month': next_moth_payment.index, 'values': next_moth_payment.values})
plt.figure(figsize=(6, 6))
plt.rcParams['font.sans-serif']=['Hiragino Sans GB'] # 设置字体，正常显示中文标签
plt.title('信用卡违约情况\n (违约： 1; 守约：0)')
sns.set_color_codes(palette='bright')
figure = sns.barplot(x='default.payment.next.month', y='values', data=df) # 绘制条形图
plt.show()
# 将图片保存
figure.figure.savefig('payment_condition.jpg')

# 特征选择
# 删除与结果无关的ID字段
data.drop(['ID'], inplace=True, axis=1)
# 分离出标注
target = data['default.payment.next.month'].values
# 分离出特征
columns = data.columns.tolist()
columns.remove('default.payment.next.month')
features = data[columns].values
# 划分训练集合（70%）、测试集（30%）; stratify 保证训练集和测试集的结果比例一致；random_state 随机数种子，保证多次划分的结果一致
train_x, test_x, train_y, test_y = train_test_split(features, target, test_size = 0.30, stratify = target, random_state = 1)

# 构造各种分类器
# 分类器中文简称
classifier_chinese_names = {
    'svc': 'SVM',
    'decisiontreeclassifier': '决策树',
    'randomforestclassifier': '随机森林',
    'kneighborsclassifier': 'KNN'
}

# 分类器名称
classifier_names = [
    'svc',
    'decisiontreeclassifier',
    'randomforestclassifier',
    'kneighborsclassifier'
]

# 构造分类器
classifier_builds = [
    SVC(random_state = 1, kernel='rbf'), # 支持向量机
    DecisionTreeClassifier(random_state=1, criterion='gini'), # CART决策树
    RandomForestClassifier(random_state=1, criterion='gini'), # 随机森林
    KNeighborsClassifier(metric='minkowski') # KNN
]

# 分类器参数
classifier_params = [
    {'svc__C':[1], 'svc__gamma':[0.01]},
    {'decisiontreeclassifier__max_depth':[6,9,11]},
    {'randomforestclassifier__n_estimators':[3,5,6]},
    {'kneighborsclassifier__n_neighbors':[4,6,8]}
]

# 使用 GridSearchCV 查找最优参数，并根据各模型的训练效果，给出准确度
def gridsearch_work(model_name, pipeline, model_params, train_x, train_y, test_x, test_y):
    result = ''
    # 使用 GridSearchCV 对模型参数进行调优
    model = GridSearchCV(estimator=pipeline, param_grid=model_params)
    res = model.fit(train_x, train_y)
    print('最优参数: ', res.best_params_)
    print('最优分数: %.4f'%res.best_score_)
    result = '最优参数: ' + str(res.best_params_) + '\n最优分数: ' + str(res.best_score_) + '\n' 
    predict_y = res.predict(test_x)
    # 用准确度对模型评估
    print(f'{classifier_chinese_names.get(model_name)}的准确度: ', accuracy_score(test_y, predict_y))
    print('---------------------')
    result += classifier_chinese_names.get(model_name) + '的准确度: ' + str(accuracy_score(test_y, predict_y)) + '\n---------------------\n'
    # 将信息保存到文件
    with open('payment_predict_info.txt', 'a') as f:
        f.write(result)

# 训练模型
for model_name, model, model_params in zip(classifier_names, classifier_builds, classifier_params):
    # 使用 pipeline 管道机制，对分类器进行流水线作业
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # 对数据进行标准化
        (model_name, model)
    ])
    # 调用定义的方法进行模型训练
    gridsearch_work(model_name, pipeline, model_params, train_x, train_y, test_x, test_y)

