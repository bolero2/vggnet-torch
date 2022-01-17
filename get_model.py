import sys
import os
import yaml
from glob import glob

from models import VGGNet


def get_model(yaml_path : str = './setting.yaml'):
    file_path = os.path.split(sys.modules[__name__].__file__)[0]
    yaml_file = os.path.join(file_path, yaml_path)

    assert os.path.isfile(yaml_file), "There isn't setting.yaml file!"
    with open(yaml_file) as f:
        setting = yaml.load(f, Loader=yaml.SafeLoader)
    assert len(setting), "either setting value must be specified. (yaml file is empty.)"

    if not isinstance(setting['classes'], list) and setting['classes'].split('.')[1] == 'txt':
        file_list = glob(f"{setting['DATASET']['root_path']}/**/{setting['classes']}", recursive=True)
        assert len(file_list) == 1, "Error."
        file_list = file_list[0]
        class_txt = open(file_list, 'r')
        classes = class_txt.readlines()
        class_txt.close()
        for i, c in enumerate(classes):
            if c[-1] == '\n':
                classes[i] = c[:-1]

        setting['classes'] = classes

    num_categories = setting['nc'] = len(setting['classes'])
    model_type = setting['network']
    setting['file_path'] = file_path

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
        name="vgg19", ch=3, num_classes=setting['nc'], setting=setting
    )

    print(network)
    exit()