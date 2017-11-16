#!/usr/bin/python
import argparse
import json
import os
import os.path as osp
import sys
import PIL.Image
import yaml
from labelme import utils
import numpy as np

def json_to_dataset(json_file_path):
    out_dir = osp.basename(json_file_path).replace('.', '_')
    out_dir = osp.join(osp.dirname(json_file_path), out_dir)
    if not osp.exists(out_dir): os.mkdir(out_dir)
    data = json.load(open(json_file_path)) 
    img = utils.img_b64_to_array(data['imageData'])
    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
    lbl_viz = utils.draw_label(lbl, img, lbl_names)
    PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
    PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label.png'))
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
    info = dict(label_names=lbl_names)
    with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
        yaml.safe_dump(info, f, default_flow_style=False)
    print('wrote data to %s' % out_dir)

def main():
    root_path = "./labelme_gen/jsons/"
    file_names = os.listdir(root_path)
    for i,file_name in enumerate(file_names,0):
        if i>=1:break
        if ".json" not in file_name: continue;
        json_to_dataset(root_path+file_name)

if __name__ == '__main__':
    main()