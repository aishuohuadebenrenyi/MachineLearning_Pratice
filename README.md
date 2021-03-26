# 机器学习算法案例实战(python实现)

## 一.[信用卡违约率分析](./信用卡违约率分析)

**1.加载数据**

用 pandas 加载[数据](./信用卡违约率分析/credit_card_info.csv)

[数据字段说明](MachineLearning_Pratice/信用卡违约率分析/数据字段说明.jpeg)

**2.数据探索性分析及可视化**

用 matplotlib 和 seaborn 对数据的标注（结果）进行[可视化](./信用卡违约率分析/payment_condition.jpg)。

**3.特征处理**

用 pandas 去除无关特征

用 StandardScaler 对数据进行标准化

**4.模型训练**

用 train_test_split 划分训练集和测试集

选择模型：SVM、决策树、随机森林、KNN

用 GridSearchCV 优化模型参数

用 Pipeline 管道机制定制化分类器训练流程

[代码](./信用卡违约率分析/credit_card_classifier.py )

**5.模型评估**

用 accuracy_score 对不同模型进行评估，[得出结论](./信用卡违约率分析/payment_predict_info.txt)。


## 二.[信用卡诈骗分析](./信用卡诈骗分析)

**1.加载数据**

用 pandas 加载数据

数据字段说明:

Time:交易时间;
Amount:交易金额;
Class:交易的分类，0表示正常（非欺诈），1表示欺诈
V1，V2，……V28：出于隐私，不知道这些特征代表的具体含义，只知道这 28 个特征值是通过 PCA 变换得到的结果


**2.数据探索性分析及可视化**

用 matplotlib 和 seaborn 对数据的标注（结果）进行[可视化](./信用卡违约率分析/payment_condition.jpg)。

**3.特征处理**

用 pandas 去除无关特征

用 StandardScaler 对数据进行标准化

**4.模型训练**

用 train_test_split 划分训练集和测试集

选择模型：LogisticRegression

[代码](./信用卡诈骗分析/creditcard.py )

**5.模型评估**

用 [混淆矩阵](./信用卡诈骗分析/混淆矩阵.png)、精确率、召回率、F1值 模型进行评估，绘制[精确率-召回率曲线](./信用卡诈骗分析/精确率_召回率曲线.png)，[得出结论](./信用卡诈骗分析/predict_info.txt)。

注：这里评价指标没有使用准确度是因为，数据分类结果严重不平衡，准确率很难反应模型的好坏。

