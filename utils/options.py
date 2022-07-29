import argparse
import yaml
from collections import OrderedDict
import random

'''written from basicsr.utils.options'''

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def parse_options(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=False, help='Path to option YAML file.')
    args = parser.parse_args()
    
    args.opt = 'test.yml'
    
    # regardless of distribute training and LLDB training
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
        
    # random seed TODO
    
    # opt['generate images']['img_ori_path'] = root_path + '/' + opt['generate images']['img_ori_path']
   
    opt['evaluated model']['arch_path'] = root_path + '/' + opt['evaluated model']['arch_path']
    opt['evaluated model']['model_path'] = root_path + '/' + opt['evaluated model']['model_path']
          
    return opt, args
    