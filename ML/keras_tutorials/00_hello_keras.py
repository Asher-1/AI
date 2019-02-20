# Keras为图片数据输入提供了一个很好的接口，即Keras.preprocessing.image.ImageDataGenerator类
# 这个类生成一个数据生成器Generator对象，依照循环批量产生对应于图像信息的多维矩阵
# 根据后台运行环境的不同，比如是TensorFlow还是Theano，多维矩阵的不同维度对应的信息
# 分别是图像二维的像素点，第三维对应于色彩通道，因此如果是灰度图像，那么色彩通道只有一个
# 维度；如果是RGB色彩，那么色彩通道有三个维度













