import numpy as np
import os,sys

arg = sys.argv[1]

with open(arg,'r') as f:
    a = f.readlines()
print(len(a))
backbone_time = [float(item.split()[-1])  for item in a if 'backbone' in item ]
backbone_time = np.array(backbone_time)

header_time = [float(item.split()[-1])  for item in a if 'header' in item ]
header_time = np.array(header_time)

converting_time = [float(item.split()[-1])  for item in a if 'box converting' in item ]
converting_time = np.array(converting_time)

nms_time = [float(item.split()[-1])  for item in a if 'nms' in item ]
nms_time = np.array(nms_time)

print('backbone_time' , backbone_time.mean())
print('header_time' , header_time.mean())
print('converting_time' , converting_time.mean())
print('nms_time' , nms_time.mean())