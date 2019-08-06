# coding:utf-8
# @Time    : 2019/1/8 10:34 PM
# @Author  : youngz
# @Email    : zhangyang@socialbird.cn
# @File    : loggers.py
# @Software: PyCharm

from __future__ import absolute_import
import os
import logging
from logging.handlers import RotatingFileHandler


class BotLoggerConfig(object):
    LOG_REL_DIR = "logs"
    path = os.path.join(os.path.dirname(__file__),"Policy")
    LOG_DIR = os.path.join(path, LOG_REL_DIR)
    DBOT_OFFLINE_LOGNAME = "dbot_offline"
    DBOT_ONLINE_LOGNAME = "dbot_online"
    DBOT_OFFLINE_LOGFILE = os.path.join(LOG_DIR, "dbot_offline.log")
    DBOT_ONLINE_LOGFILE = os.path.join(LOG_DIR, "dbot_online.log")
    OFFLINE_LOG_LEVEL = logging.INFO
    ONLINE_LOG_LEVEL = logging.INFO
    pass


def setup_logger(logger_name, log_file, level=logging.DEBUG):
    l = logging.getLogger(logger_name)

    fmt = "[%(asctime)-15s] [%(levelname)s] %(filename)s [line=%(lineno)d] [PID=%(process)d] %(message)s"
    datefmt = "%a %d %b %Y %H:%M:%S"

    formatter = logging.Formatter(fmt, datefmt)

    fileHandler = RotatingFileHandler(filename=log_file, mode='a', maxBytes=1024 * 1024 * 5, backupCount=5,
                             encoding='utf-8')  # 使用RotatingFileHandler类，滚动备份日志

    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)
    pass


setup_logger(BotLoggerConfig.DBOT_OFFLINE_LOGNAME, BotLoggerConfig.DBOT_OFFLINE_LOGFILE)
setup_logger(BotLoggerConfig.DBOT_ONLINE_LOGNAME, BotLoggerConfig.DBOT_ONLINE_LOGFILE)

dbot_offline_logger = logging.getLogger(BotLoggerConfig.DBOT_OFFLINE_LOGNAME)
dbot_online_logger = logging.getLogger(BotLoggerConfig.DBOT_ONLINE_LOGNAME)

