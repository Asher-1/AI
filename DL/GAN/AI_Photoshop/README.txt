“案例二AI制图训练结果”文件夹里包含我训练得到的参数文件 generator_weight，
可直接用于生成图片。

images 文件夹里的是训练所用的图片。

=======
训练：

运行 python train.py 即可开始训练。

默认训练 100 个 Epoch，你可以提早结束训练（Ctrl + C）。

=======
生成图片：

训练完很多个 Epoch （训练几个 Epoch 是不够的，图片比较模糊）之后，
运行 python generate.py 即可生成图片。

如果等不了那么久，可以用我提供在“案例二AI制图训练结果”文件夹里的参数文件。
使 generator_weight 参数文件位于 generate.py 同级目录下，
运行 python generate.py 即可生成图片。

==============
大家好，我是 谢恩铭（慕课网 imooc.com 的 Oscar老师）：
http://www.imooc.com/u/1289460
非常感谢你选择我的课程，祝学习顺利。

这是我在慕课网的《基于Python玩转人工智能最火框架TensorFlow应用实践》
https://coding.imooc.com/class/176.html
这门实战课程的代码和素材，是我和慕课网的共有知识产权，只限讲师、慕课网教学和学员使用，
禁止非慕课网的一切商用行为。谢谢！

==============
我的简书：
https://www.jianshu.com/u/44339a8a9afa
我的微信：frogoscar
我的 QQ：379641629
我的邮箱：enmingx@gmail.com
欢迎提问、交流、互相学习。
