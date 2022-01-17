import sys
import os
import yaml
import json
import glob

from models import VGGNet


def get_model(yaml_path : str = './setting.yaml'):
    file_path = os.path.split(sys.modules[__name__].__file__)[0]

    yaml_file = os.path.join(file_path, yaml_path)

    assert os.path.isfile(yaml_file), "There isn't setting.yaml file!"
    with open(yaml_file) as f:
        setting = yaml.load(f, Loader=yaml.SafeLoader)
    assert len(setting), "either setting value must be specified. (yaml file is empty.)"
    
    num_categories = setting['nc']
    category_names = setting['classes']
    model_type = setting['network']

    network = VGGNet(
        name=model_type, ch=3, num_classes=num_categories, setting=setting
    )

    return network


if __name__ == "__main__":
    setting = {
        "img_size": [224, 224, 3],
        "nc": 1,
        "classes": ['cat'],
        "DATASET": {
            "root_path": "./",
            "ext": "jpg"
        },
        "vgg19": [
            [64, 64],
            [128, 128], 
            [256, 256, 256, 256], 
            [512, 512, 512, 512],
            [512, 512, 512, 512]
        ],
        "fc_layer":  [4096, 4096],
        "workers": 4
    }

    network = VGGNet(
        name="vgg19", ch=3, num_classes=20, setting=setting
    )

    print(network)
    exit()