import requests
import os
import base64
import zipfile
import inspect
from pathlib import Path
from loguru import logger
from utils import GenerateFile, read_yaml, set_time_limit, get_date, create_name, create_dir, write_yaml

__dir__ = Path(__file__).parent

GenerateFile(
    os.path.join(__dir__, "conf"),
    os.path.join(__dir__, "conf/template/locl.yaml")
)



class StableDiffusionApi:
    OUTPUT_PATH = f"{__dir__}/outputs"
    URLS = read_yaml(f"{__dir__}/conf/urls.yml")
    # OUT_TIME = 3600

    def __init__(self):
        """Init"""
        logger.debug(f"load URL Config: {self.URLS}")
        self.load_config()

    # @set_time_limit(OUT_TIME, sys._getframe())
    def __post__(self, url, rdata=None, jdata=None):
        """POST请求"""
        response = None
        try:
            response = requests.post(url=url, data=rdata, json=jdata)
        except:
            logger.error("POST error")
        return response

    def load_config(self):
        """加载配置文件"""
        self.conf = dict()
        for root, _, file_name in os.walk(os.path.join(__dir__, "conf/built_in")):
            for name in file_name:
                n, s = name.split(".")
                if s != "yml":
                    continue
                self.conf[n] = read_yaml(os.path.join(root, name))

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

    def img2img(self, img_base64_list: list, **args):
        """图生图
        
        img_base64_list (list[base64]): 图片的base64编码列表
        **args: 其余参数，参照/conf/built_in对应的yml文件进行设置, 新增参数需要在其文件中设置
        """
        # 获取参数信息
        try:
            data = self.conf.get(inspect.stack()[0][3])
        except:
            logger.error("img2img 未设置对应的yml文件")

        data["init_images"] = img_base64_list
        for k, v in args.items():
            if k in data:
                data[k] = v
            else:
                logger.info(f"img2img 的默认参数中, 是否需要增加参数: {k} = {v}")
        response = self.__post__(self.URLS["img2img"], jdata=data)

        # 检测图片存储目录，没有就创建
        path = f"{__dir__}/img2img/" + "/".join(get_date())
        create_dir(path)
        # 存储输入图片
        self.file_store(img_base64_list, path)

        # 创建生成图片文件夹，并进行打包
        generate_path = f"{self.OUTPUT_PATH}/{create_name(flag=True)}"
        create_dir(generate_path)
        self.file_store(response.json()["images"], path=generate_path, pack=True)
        return f"{generate_path}.zip"
        
    def high_definition(self, img_base64_list: list, **args):
        """高清化"""
        try:
            data = self.conf.get(inspect.stack()[0][3])
        except:
            logger.error("high_definition 未设置对应的yml文件")

        data["imageList"] = img_base64_list

        for k, v in args.items():
            if k in data:
                data[k] = v
            else:
                logger.info(f"high_definition 的默认参数中, 是否需要增加参数: {k} = {v}")
        logger.debug(data.keys())
        response = self.__post__(self.URLS["high_definition"], jdata=data)

        # 检测图片存储目录，没有就创建
        path = f"{__dir__}/high_definition/" + "/".join(get_date())
        create_dir(path)
        # 存储输入图片
        self.file_store(img_base64_list, path)

        # 创建生成图片文件夹，并进行打包
        generate_path = f"{self.OUTPUT_PATH}/{create_name(flag=True)}"
        create_dir(generate_path)
        self.file_store(response.json()["images"], path=generate_path, pack=True)
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





sd = StableDiffusionApi()

with open("./01.png", "rb") as f:
    png = f.read()

sd.high_definition(img_base64_list=[
    {"data": base64.b64encode(png).decode()
     ,"name": "a.png"}
])
# sd.img2img(img_base64_list=[base64.b64encode(png).decode()])
# write_yaml("setting.yaml", sd.get_sd_config().json())

# data = {
#     "init_images": "img_base64_list",
#     "steps": 10,
#     "cfg_scale": 8,
#     "width": 768,
#     "height": 1024,
#     "n_iter": 2,
#     "sampler_index": "Euler"
# }
# write_yaml("t.yaml", data)


# import inspect
# def foo(a, **args):
#     print(a)
#     for k, v in args.items():
#         print([k, v])



# foo(a=1, b=2, c=3)