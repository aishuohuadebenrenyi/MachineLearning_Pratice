# 机器学习算法案例实战(python实现)

## 一.[信用卡违约率分析](https://github.com/aishuohuadebenrenyi/MachineLearning_Pratice/tree/main/%E4%BF%A1%E7%94%A8%E5%8D%A1%E8%BF%9D%E7%BA%A6%E7%8E%87%E5%88%86%E6%9E%90)

**1.加载数据**

用 pandas 加载[数据](./信用卡违约率分析/credit_card_info.csv)

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



