import os

import yaml


def general_config():

    file_config = open('resources/general_properties.yaml', encoding='utf8')
    conf = yaml.load(file_config, Loader=yaml.FullLoader)
    return conf


def secret_config():

    file_config = open('resources/secret_properties.yaml', encoding='utf8')
    conf = yaml.load(file_config, Loader=yaml.FullLoader)
    return conf
