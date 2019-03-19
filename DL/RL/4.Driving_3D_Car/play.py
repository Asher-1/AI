# -*- coding: UTF-8 -*-

"""
主程序
1. 执行命令：包括创建指定的目录，调用 worker.py 里的程序
来创建 ps 和 worker 进程，以及创建 TensorBoard 进程
2. 打印执行的命令到屏幕
3. 把一些命令写入脚本文件，以便用 kill 命令来结束进程和
用 tail 命令来查看进程的输出

== 分布式（Distributed） TensorFlow 概念 ==
分布式 TensorFlow 底层的通信是 gRPC (google Remote Procedure Call)
RPC 是 远程过程调用的缩写。

cluster ：集群。是 Job 的集合
ps ：Parameter Server（参数服务器），管理参数的存储和更新工作
worker ：运行操作。会从 ps 拉取参数，及向 ps 推送参数
task ：主机上的一个进程。多数情况下,一个机器上只运行一个 task
Job ：是Task的集合

分布式深度学习框架中,我们一般把 Job 划分为 ps（Parameter Server，参数服务器）
和 worker（工作者）。ps 是管理参数的存储和更新工作；worker 是来运行操作。
如果参数的数量太大，一台机器处理不了，就需要多个 task
"""

from six.moves import shlex_quote
import os
import sys
import argparse

# 读取参数
parser = argparse.ArgumentParser(description="命令的参数")
parser.add_argument('-w', '--num-workers', default=1, type=int,
                    help='worker 的数目')
parser.add_argument('-r', '--remotes', default=None,
                    help='远程 VNC 服务器的地址 (例如：-r vnc://localhost:5900+15900, vnc://localhost:5901+15901).')
parser.add_argument('-l', '--log-dir', type=str, default="neonrace",
                    help='包含日志文件和各种 checkpoint（检查点）文件的总目录')
parser.add_argument('--visualise', action='store_true',
                    help='可视化与否：如果设置，那么在每个时间步之间用 env.render() 来渲染（显示）游戏环境')


# 每一句被创建的新命令
def new_cmd(session, name, cmd, logdir, shell):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shlex_quote(str(v)) for v in cmd)

    # nohup 的模式来运行命令
    # nohup 命令可以将程序以忽略挂起信号（hang-up）的方式运行起来
    return name, "nohup {} -c {} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(shell, shlex_quote(cmd), logdir, session, name, logdir)


# 创建命令
def create_commands(session, num_workers, remotes, logdir, shell='bash', visualise=False):
    # 启动各个 Worker 和 启动 TensorBoard
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=',
        sys.executable, 'worker.py',
        '--log-dir', logdir,
        '--num-workers', str(num_workers)]

    # 如果需要可视化
    if visualise:
        base_cmd += ['--visualise']

    if remotes is None:
        remotes = ["1"] * num_workers
    else:
        remotes = remotes.split(',')
        assert len(remotes) == num_workers

    cmds_map = [new_cmd(session, "ps", base_cmd + ["--job-name", "ps"], logdir, shell)]

    # 对每一个 worker
    for i in range(num_workers):
        cmds_map += [new_cmd(session,
            "w-%d" % i, base_cmd + ["--job-name", "worker", "--task", str(i), "--remotes", remotes[i]], logdir, shell)]

    cmds_map += [new_cmd(session, "tb", ["tensorboard", "--logdir", logdir, "--port", "6006"], logdir, shell)]

    notes = []
    cmds = [
        "mkdir -p {}".format(logdir),
        "echo {} {} > {}/cmd.sh".format(sys.executable, ' '.join([shlex_quote(arg) for arg in sys.argv if arg != '-n']), logdir),
    ]
    cmds += ["echo '#!/bin/sh' >{}/kill.sh".format(logdir)]
    notes += ["运行 `source {}/kill.sh` 命令来结束各个进程".format(logdir)]

    notes += ["运行 `tail -f {}/*.out` 命令来查看各个进程的输出".format(logdir)]
    notes += ["在浏览器中打开 http://localhost:6006 ，查看 TensorBoard 运行的结果"]

    for window, cmd in cmds_map:
        cmds += [cmd]

    return cmds, notes


def run():
    args = parser.parse_args()
    cmds, notes = create_commands("a3c", args.num_workers, args.remotes, args.log_dir, visualise=args.visualise)

    print("\n运行以下命令:")
    print("\n".join(cmds))
    print("")
    os.system("\n".join(cmds))
    print('\n'.join(notes))


if __name__ == "__main__":
    run()
