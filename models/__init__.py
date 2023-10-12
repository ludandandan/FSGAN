import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):# model_name:apdrawing_gan
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename) # 载入apdrawing_gan_model.py

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name) #根据模型的名字找到模型的类
    return model_class.modify_commandline_options


def create_model(opt):
    model = find_model_using_name(opt.model) # 根据模型的名字找到模型类
    instance = model()#定义模型对象
    instance.initialize(opt) # 用命令行参数初始化模型
    print("model [%s] was created" % (instance.name()))
    return instance
