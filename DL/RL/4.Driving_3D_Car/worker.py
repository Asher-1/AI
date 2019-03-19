# -*- coding: UTF-8 -*-

"""
分布式集群（Cluster）的创建和配置，等
"""

import os
import cv2
import sys
import time
import signal
import logging
import argparse
import go_vncdriver
import tensorflow as tf

from a3c import A3C
from env import create_env

# 配置日志系统
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 设置 write_meta_graph 为 False（不保存），因为很耗时，可能会拖慢训练
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename, meta_graph_suffix, False)


# 如果是 worker 的话，所运行的步骤
def run(args, server):
    env = create_env(client_id=str(args.task), remotes=args.remotes)
    trainer = A3C(env, args.task, args.visualise)

    # 以 'local' 开头的变量（局部变量）不会被保存在 checkpoint 参数文件中
    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()

    # 保存变量到参数文件中
    saver = FastSaver(variables_to_save)

    # 获取可被训练的变量
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    logger.info('可被训练的变量 :')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
        logger.info("初始化所有参数。")
        ses.run(init_all_op)

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])

    logdir = os.path.join(args.log_dir, 'train')
    # 写入 TensorBoard 的日志文件
    summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)

    logger.info("存储 TensorBoard 文件的目录: %s_%s", logdir, args.task)

    # 一个高层的 Wrapper（包装类）
    # 可以做 TensorBoard 日志文件的保存，参数文件的保存，等等操作
    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,  # 存储参数文件的目录
                             saver=saver,    # 存储参数文件所用的 Saver
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,  # 存储 TensorBoard 日志文件的 FileWriter
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    # 总的可运行步数。可修改
    num_global_steps = 100000000

    logger.info("启动会话中...")
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        sess.run(trainer.sync)
        trainer.start(sess, summary_writer)
        global_step = sess.run(trainer.global_step)
        logger.info("在第 %d 步开始训练", global_step)
        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            trainer.process(sess)
            global_step = sess.run(trainer.global_step)

    # 停止所有服务
    sv.stop()
    logger.info('已经 %s 步了. worker 被停止.', global_step)


# 集群（Cluster）的配置
def cluster_spec(num_workers, num_ps):
    cluster = {}
    port = 12222

    # 所有 ps（参数服务器）的配置
    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    # 所有 worker 的配置
    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers

    return cluster


# 主函数
def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='设置冗余程度')
    parser.add_argument('--task', default=0, type=int, help='task 下标')
    parser.add_argument('--job-name', default="worker", help='Job 是 worker 还是 ps')
    parser.add_argument('--num-workers', default=1, type=int, help='worker 的数目')
    parser.add_argument('--log-dir', default="neonrace", help='包含日志文件和各种 checkpoint（检查点）文件的总目录')
    parser.add_argument('-r', '--remotes', default=None,
                        help='远程 VNC 服务器的地址 (例如：-r vnc://localhost:5900+15900, vnc://localhost:5901+15901).')
    parser.add_argument('--visualise', action='store_true',
                        help='可视化与否：如果设置，那么在每个时间步之间用 env.render() 来渲染（显示）游戏环境')

    args = parser.parse_args()
    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    # 定义终止线程的信号
    def shutdown(signal):
        logger.warn('收到信号 %s: 退出中...', signal)
        sys.exit(128+signal)
    # hang-up 信号，例如用户注销（logout）
    # 或者 网络断开时（hang-up 信号 会被 nohup 模式忽略）
    signal.signal(signal.SIGHUP, shutdown)
    # interrupt 信号，例如 Ctrl + C 组合键
    signal.signal(signal.SIGINT, shutdown)
    # terminal 信号。例如 kill 命令
    signal.signal(signal.SIGTERM, shutdown)

    # 如果 Job 是 worker
    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task, config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run(args, server)
    # 如果 Job 是 ps（Parameter Server）
    else:
        tf.train.Server(cluster, job_name="ps", task_index=args.task, config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)


if __name__ == "__main__":
    # 运行 main()
    tf.app.run()
