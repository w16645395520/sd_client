import requests
import yaml
from loguru import logger
from utils import read_yaml, set_time_limit
import sys
import base64


class StableDiffusionApi:
    URLS = read_yaml("./config.yaml")
    OUT_TIME = 3600
    def __init__(self):
        """Init"""
        logger.debug(f"load URL Config: {self.URLS}")

    @set_time_limit(OUT_TIME, sys._getframe())
    def __post__(self, url, rdata=None, jdata=None):
        """POST请求"""
        response = None
        try:
            response = requests.post(url=url, data=rdata, json=jdata)
        except:
            logger.error("POST error")
        return response

    @set_time_limit(OUT_TIME, sys._getframe())
    def __get__(self, url, header=None):
        response = None
        try:
            response = requests.get(url)
        except:
            logger.error("GET error")
        return response

    def get_sd_appid(self):
        """获取sd的appid标识"""
        response = self.__get__(self.URLS["app_id"])
        return response

    def refresh_checkpoints(self):
        """刷新模型数据"""
        response = self.__post__(self.URLS["refresh_checkpoints"])
        return
    
    def reload_checkpoint(self):
        """重载模型数据"""
        response = self.__post__(self.URLS["reload_checkpoint"])
        return

    def unload_checkpoint(self):
        """卸载模型数据"""
        response = self.__post__(self.URLS["unload_checkpoint"])
        return

    def get_sd_config(self):
        """获取模型参数信息"""
        response = self.__get__(self.URLS["get_config"])
        return response

    def set_sd_config(self, **args):
        """设置（修改）模型参数信息"""
        # 支持修改所有的get_sd_config获取的所有参数
        response = self.__post__(self.URLS["set_config"])
    
    def txt2img(self, **arg):
        """文生图"""
        data = {
            "prompt": "best quality, ultra high res, (photorealistic:1.4), 1girl, brown blazer, black skirt, glasses, thighhighs, ((T shirt)), (upper body), (Kpop idol), (aegyo sal:1), (platinum   blonde hair:1), ((puffy eyes)), looking at viewer, facing front, smiling",
            "negative_prompt": "nsfw, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, age spot, glan",
            "steps": 10,
            "cfg_scale": 8,
            "width": 768,
            "height": 1024,
            "n_iter": 1,
            "sampler_index": "Euler"
        }
        # response.json().keys() = ['images', 'parameters', 'info']
        response = self.__post__(self.URLS["txt2img"], jdata=data)
        
    def img2img(self, **arg):
        """图生图"""
        with open("./01.png", "rb") as f:
            png = f.read()
        
        data = {
            "init_images": [base64.b64encode(png).decode()],
            "steps": 10,
            "cfg_scale": 8,
            "width": 768,
            "height": 1024,
            "n_iter": 1,
            "sampler_index": "Euler"
        }
        response = self.__post__(self.URLS["img2img"], jdata=data)
        response = requests.post(self.URLS["img2img"], json=data)
        print(response.json())
        


        

sd = StableDiffusionApi()
sd.img2img(a=1)

# 获取内置信息
# response = requests.get(config["get_config"])
# print(response.json())

# # 设置模型
# response = requests.post(config["set_config"], json={"sd_model_checkpoint": "f222.safetensors", "samples_save": True})
# print(response.json())

# response = requests.get(config["get_config"])
# print(response.json())

