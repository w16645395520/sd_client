import requests
from loguru import logger
from utils import GenerateFile, read_yaml, set_time_limit, get_date, create_name, create_dir, write_yaml
import sys
import os
import base64
import zipfile


class StableDiffusionApi:
    INPUT_PATH = "/root/autodl-tmp/program/src"
    OUTPUT_PATH = "/root/autodl-tmp/program/src/outputs"
    URLS = read_yaml(f"{INPUT_PATH}/conf/urls.yml")
    # OUT_TIME = 3600

    def __init__(self):
        """Init"""
        logger.debug(f"load URL Config: {self.URLS}")

    # @set_time_limit(OUT_TIME, sys._getframe())
    def __post__(self, url, rdata=None, jdata=None):
        """POST请求"""
        response = None
        try:
            response = requests.post(url=url, data=rdata, json=jdata)
        except:
            logger.error("POST error")
        return response

    # @set_time_limit(OUT_TIME, sys._getframe())
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
        response = self.__post__(self.URLS["set_config"], jdata=args)
    
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
        response = self.__post__(self.URLS["txt2img"], jdata=data)
        
        # 创建生成图片文件夹，并进行打包
        generate_path = f"{self.OUTPUT_PATH}/{create_name(flag=True)}"
        create_dir(generate_path)
        self.file_store(response.json()["images"], path=generate_path, pack=True)
        # zip文件路径 generate_path + ".zip"
        return f"{generate_path}.zip"

    def img2img(self, img_base64_list: list):
        """图生图"""
        # 检测图片存储目录，没有就创建
        path = f"{self.INPUT_PATH}/img2img/" + "/".join(get_date())
        create_dir(path)

        # 存储输入图片
        self.file_store(img_base64_list, path)

        data = {
            "init_images": img_base64_list,
            "steps": 10,
            "cfg_scale": 8,
            "width": 768,
            "height": 1024,
            "n_iter": 2,
            "sampler_index": "Euler"
        }
        response = self.__post__(self.URLS["img2img"], jdata=data)

        # 创建生成图片文件夹，并进行打包
        generate_path = f"{self.OUTPUT_PATH}/{create_name(flag=True)}"
        create_dir(generate_path)
        self.file_store(response.json()["images"], path=generate_path, pack=True)
        # zip文件路径 generate_path + ".zip"
        return f"{generate_path}.zip"
        

    def file_store(self, images: list[bytes]=None, path=None, pack=False):
        """文件存储、打包"""
        if pack:
            # 打包
            zip_obj = zipfile.ZipFile(path + ".zip", "w")
        try:
            for img in images:
                img1 = base64.b64decode(img.encode())
                # 处理后缀问题
                t = str(img1[:20]).split("\\r")[0].lower()
                sfx = "png"
                for sfx in ["png", "jpeg"]:
                    if sfx in t:
                        break
                # 图片文件名称
                file_name = create_name(img1) + f".{sfx}"
                file_path = os.path.join(path, file_name)
                with open(file_path, "wb") as f:
                    f.write(img1)
                # 写入压缩文件中
                if pack:
                    zip_obj.write(file_path, file_name)
        finally:
            if pack:
                zip_obj.close()





# sd = StableDiffusionApi()

# with open("./01.png", "rb") as f:
#     png = f.read()

# # sd.img2img(img_base64_list=[base64.b64encode(png).decode()])
# sd.get_sd_config()

