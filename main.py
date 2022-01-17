from get_model import get_model
from glob import glob


if __name__ == "__main__":
    model = get_model(yaml_path='setting.yaml')

    imagelist = glob(f"{model.root_dir}/**/*.{model.ext}", recursive=True)
    print(imagelist)

    # model.fit()