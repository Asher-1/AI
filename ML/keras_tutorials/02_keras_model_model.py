# 通用模型
# 通用模型可以用来设计非常复杂、任意拓扑结构的神经网络，例如有向无环图网络
# 类似于序列模型，通用模型通过函数化的应用接口来定义模型
# 使用函数化的应用接口有好多好处，比如：决定函数执行结果的唯一要素是其返回值，而决定
# 返回值的唯一要素则是其参数，这大大减轻了代码测试的工作量

# 在通用模型中，定义的时候，从输入的多维矩阵开始，然后定义各层及其要素，最后定义输出层
# 将输入层和输出层作为参数纳入通用模型中就可以定义一个模型对象

from keras.layers import Input
from keras.layers import Dense
from keras.models import Model

# 定义输入层
input = Input(shape=(784,))
# 定义各个连接层，假设从输入层开始，定义两个隐含层，都有64个神经元，都使用relu激活函数
x = Dense(64, activation='relu')(input)
x = Dense(64, activation='relu')(x)
# 定义输出层，使用最近的隐含层作为参数
y = Dense(10, activation='softmax')(x)

# 所有要素都齐备以后，就可以定义模型对象了，参数很简单，分别是输入和输出，其中包含了
# 中间的各种信息
model = Model(inputs=input, outputs=y)

# 当模型对象定义完成之后，就可以进行编译了，并对数据进行拟合，拟合的时候也有两个参数
# 分别对应于输入和输出
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(data, labels)




