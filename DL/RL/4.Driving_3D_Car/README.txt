首先确保进入我们课程里带大家配置的 universe 这个 Anaconda 的虚拟环境。
可以参看我在简书的文章：
《Windows,Linux,macOS三平台安装OpenAI的Gym和Universe》
https://www.jianshu.com/p/536d300a397e

当然了，如果你自己会配置，那么自己配置 Gym 和 Universe 开发环境也可以
（也许就不需要下面这句 source activate universe 命令了）。

================
运行游戏：

运行 source activate universe ，进入 universe 这个 Anaconda 的虚拟环境，
然后 加上参数运行 python play.py 即可，例如：

python play.py --num-workers 2 --log-dir neonrace

--num-workers 2 表示有两个 worker（类似 A3C 中的客户端），
你也可以指定多个，比如 3，4，8，16，等等，视你的 CPU 或 GPU 核心数目而定。
--log-dir neonrace 表示将参数文件和 TensorBoard 需要
的日志文件都放入当前目录下叫 neonrace 的文件夹（不存在的话会被创建）。

================
要看 3D 赛车在 NeonRace 环境里实时情况：

如果在 Windows 或 Ubuntu 下，请打开浏览器，在地址栏输入：
http://localhost:15900/viewer/?password=openai （表示第 1 个 worker 的环境）
或
http://localhost:15901/viewer/?password=openai （表示第 2 个 worker 的环境）
如果你开了超过 2 个 worker（），地址依次类推。

如果在苹果的 macOS 下，则比较方便，不需要打开浏览器输入地址，
macOS 自带了一个 VNC 的相关软件。
在终端里输入：
open vnc://localhost:5900 (连接密码是 openai 。表示第 1 个 worker 的环境)
或
open vnc://localhost:5901 (连接密码是 openai 。表示第 2 个 worker 的环境)
如果你开了超过 2 个 worker，地址依次类推，
例如 5902（表示第 3 个 worker），5903（表示第 3 个 worker），等等。

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
