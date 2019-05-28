1. 代码运行环境
Ubuntu16.04 包含的库: Python3.5    keras2.2.0.  numpy cv2

2. 实现思路
首先利用已有的imagenet上预训练好的模型得到所有训练图片的feature map，然后根据论文当中求均值和方差的公式计算出均值以及方差矩阵，从而可以获得所有训练集样本上的最大特征值所对应的特征向量。然后根据该特征向量可以求得测试样本的P矩阵，经过上采样回原图大小之后，以0为阈值即可得到DDT算法给出的测试图片的物体像素信息，最后利用联通算法即可求的物体位置信息。

3. 对DDT算法的理解
DDT算法相当于利用PCA求得最能体现所需检测物体的特征的特征向量，利用该特征向量在原图中高亮物体，从而可以得到物体的目标框。为了保证求得最好的特征向量，因而需要保持训练数据一定的纯度，即大多数训练数据都需要为同一种类别。在实际使用过程中，发现对于有复杂背景的图片不能较好的得到目标框，对于多目标检测效果也不好。

4. 效果展示
![](https://github.com/Ezereal/DDT/blob/master/data/car_result/0.jpg)
![](https://github.com/Ezereal/DDT/blob/master/data/car_result/2.jpg)
![](https://github.com/Ezereal/DDT/blob/master/data/car_result/3.jpg)
![](https://github.com/Ezereal/DDT/blob/master/data/tiger_result/2_featurevec/8.jpg)
![](https://github.com/Ezereal/DDT/blob/master/data/tiger_result/2_featurevec/15.jpg)
