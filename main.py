from get_model import get_model
from random import shuffle
import os
from glob import glob


if __name__ == "__main__":
    model = get_model(yaml_path='setting.yaml')

    imagelist = glob(f"{model.root_dir}/**/*.{model.ext}", recursive=True)
    count = len(imagelist)
    shuffle(imagelist)

    train_rate, valid_rate, test_rate = 0.8, 0.1, 0.1

    train_bundle = imagelist[:int(count * train_rate)]
    valid_bundle = imagelist[int(count * train_rate):int(count * (train_rate + valid_rate))]
    test_bundle = imagelist[int(count * (train_rate + valid_rate)):]

    print(" Train: {} | Valid: {} | Test: {}".format(len(train_bundle), len(valid_bundle), len(test_bundle)))

    trainset, validset, testset = [[], []], [[], []], [[], []]          # [ [image-list], [label-list] ]

    for elem in train_bundle:
        labelname = os.path.dirname(elem).split('/')[-1]
        trainset[0].append(elem)
        trainset[1].append(labelname)

    for elem in valid_bundle:
        labelname = os.path.dirname(elem).split('/')[-1]
        validset[0].append(elem)
        validset[1].append(labelname)

    for elem in test_bundle:
        labelname = os.path.dirname(elem).split('/')[-1]
        testset[0].append(elem)
        testset[1].append(labelname)

    model.fit(x=trainset[0],
              y=trainset[1],
              validation_data=validset,
              epochs=30,
              batch_size=4)