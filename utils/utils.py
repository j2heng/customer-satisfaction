import os
import time 
import json
from dotmap import DotMap


def create_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print ("Creation of the directory %s failed" % dir)
        exit(-1)
    else:
        print ("Successfully created the directory %s " % dir)


def get_configs(path):
    with open(path, "r") as f:
        config_dict = json.load(f)
    # convert the dictionary to a namespace 
    config = DotMap(config_dict)
    config.callbacks.checkpoint_subdir = os.path.join(config.callbacks.checkpoint_dir, 
                            time.strftime("%Y-%m-%d/",time.localtime()))
    if not os.path.exists(config.callbacks.checkpoint_subdir):
        create_dir(config.callbacks.checkpoint_subdir)
    return config 