#encoding=utf-8
import os
import yaml
import time
import datetime
from functools import wraps
from threading import Thread
from loguru import logger
from jinja2 import Environment, FileSystemLoader
import hashlib
import json


def read_yaml(path):
    """获取yaml内容"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(path, content: json):
    """写入yaml文件"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, indent=4, allow_unicode=True)

def set_time_limit(t, a):
    def auto_quit(t1, a):
        """此为控制进程超时退出的线程函数"""
        time.sleep(t1)
        logger.error(f"运行超时: {t1} {a}")
        os._exit(1) # 此函数专门用于线程控制主进程退出，有兴趣的可以看一下和sys.exit()的区别
    def decorator(f):
        """此函数用于传入被装饰函数f"""
        @wraps(f)
        def wrapper(*args,**kwargs):
            """装饰器内部遵循的逻辑是：

            1.auto_quit先执行完，进程结束
            2.被修饰函数f先执行完，auto_quit函数停止执行
            3.被修饰函数执行完，下面的代码才能运行
            """
            t1=Thread(target=auto_quit,args=(t, a))  # 此处的t是set_time_limit函数的形参，是auto_quit函数的实参
            t2=Thread(target=f,args=args,kwargs=kwargs)
            t1.setDaemon(True) # 满足第2点
            t1.start()
            t2.start()
            t2.join() # 满足第3点
        return wrapper
    return decorator


def get_date():
    current_time = datetime.datetime.now()
    result = str(current_time).split()[0].split("-")
    return result


def create_dir(path="/root/autodl-tmp/program/src"):
    if os.path.isdir(path):
        return True
    try:
        os.makedirs(path)
    except FileExistsError as e:
        logger.error(f"Owned Path: {path}")
    return True


def create_name(content = None, flag: bool = False):
    """生成名字"""
    if flag:
        # 时间编码
        return hashlib.md5(str(time.time()).encode("utf-8")).hexdigest()
    # 内容编码
    if isinstance(content, str) and content is not None:
        result = hashlib.md5(content.encode("utf-8")).hexdigest()
    elif isinstance(content, bytes) and content is not None:
        result = hashlib.md5(content).hexdigest()
    else:
        logger.error(f"content[:10]: {content[:10]}; flag: {flag}")
        raise
    return result



# #利用jinja2框架生成配置文件
# # load data from yaml file
# config = yaml.safe_load(open(args.data, 'r', encoding='utf-8'))

# # load jinja2 template
# env = Environment(loader = FileSystemLoader(args.root, encoding='utf-8'), trim_blocks=True, lstrip_blocks=True)

# def generateFile(src, dest):
#     '''
#     根据模板文件生成结果
#     @param src: 模板文件名
#     @param dest: 目标文件名
#     '''
#     template = env.get_template(src)

#     with open(dest, 'w', encoding='utf-8') as f:
#         print('# AUTO GENERATED BY JINJA2', file=f)
#         print(template.render(config), file=f)

#     print('[{}] generate [{}] by jinja2 DONE'.format(src, dest))

# # find all j2 template file under ../
# for root, dirs, files in os.walk(args.root):
#     for filename in files:
#         if filename in args.exclude: continue
#         if args.file is not None and filename != args.file: continue

#         name, ext = os.path.splitext(filename)

#         if ext != '.j2': continue

#         # 模板文件名
#         templateFilename = os.path.relpath(os.path.join(root, filename), args.root)
#         # 新的文件名
#         newFilename = os.path.join(root, name)

#         generateFile(templateFilename, newFilename)
