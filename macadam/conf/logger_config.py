# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/7/28 0:24
# @author   :Mo
# @function :logger of Macadam


from macadam.conf.path_config import path_root, path_model_dir
from logging.handlers import RotatingFileHandler
import logging
import time
import os


# log目录地址
path_logs = os.path.join(path_root, "logs")
if not os.path.exists(path_logs):
    os.mkdir(path_logs)
# 全局日志格式
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# 定义一个日志记录器
logger = logging.getLogger("Macadam")
logger.setLevel(level = logging.INFO)
# 日志文件名,为启动时的日期
# log_file_name = time.strftime("%Y-%m-%d", time.localtime(time.time())) + ".log"
log_file_name = "macadam.log"
log_name_day = os.path.join(path_logs, log_file_name)
# 文件输出, 定义一个RotatingFileHandler，最多备份3个日志文件，每个日志文件最大32K
rHandler = RotatingFileHandler(log_name_day, maxBytes=32*1024, backupCount=3, encoding="utf-8")
rHandler.setLevel(logging.INFO)
# 日志输出格式
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
rHandler.setFormatter(formatter)
# # 控制台输出
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# console.setFormatter(formatter)
# logger加到handel里边
logger.addHandler(rHandler)
# logger.addHandler(console)


# 所有文件共用一个logger
def get_logger_root(name="Macadam"):
    return logging.getLogger(name)


if __name__ == "__main__":
    logger = get_logger_root("Macadam")
    logger.info("test")

