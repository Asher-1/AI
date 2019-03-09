data 文件夹里的是训练和测试所用的 PTB 数据集。

=======
训练：

运行 python train.py 即可开始训练。
也可以使用参数 --data_path 来指定所使用的数据集的目录，
一般来说使用我提供的 data 目录即可，不需要用 --data_path 参数。

=======
测试：

训练完很多个 Epoch（最终精度达到大概 30% 以上）之后，
将训练时生成的 train-checkpoint-* 参数文件之一改名
（名字匹配 utils.py 中的 load_file），
运行 python generate.py 即可生成图片。

也可以使用参数 --load_file 来指定要用哪个参数文件来测试，
例如：
python generate.py --load_file train-checkpoint-67

具体使用请看本课程的视频和参考 utils.py 文件。

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
