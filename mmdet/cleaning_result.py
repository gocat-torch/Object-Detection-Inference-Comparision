import numpy as np
import os,sys

arg = sys.argv[1]

with open(arg,'r') as f:
    a = f.readlines()
print(len(a))
backbone_time = [float(item.split()[-1])  for item in a if 'backbone' in item ]
backbone_time = np.array(backbone_time)

neck_time = [float(item.split()[-1])  for item in a if 'neck' in item ]
neck_time = np.array(neck_time)

rpn_time = [float(item.split()[-1])  for item in a if 'rpn time' in item ]
rpn_time = np.array(rpn_time)

rpn_post_time = [float(item.split()[-1])  for item in a if 'rpn post-process' in item ]
rpn_post_time = np.array(rpn_post_time)

roi_time = [float(item.split()[-1])  for item in a if 'roi extractor' in item ]
roi_time = np.array(roi_time)


head_time = [float(item.split()[-1])  for item in a if 'head time' in item ]
head_time = np.array(head_time)

head_post_time = [float(item.split()[-1])  for item in a if 'head post-process' in item ]
head_post_time = np.array(head_post_time)


assert len(rpn_time)  == len(rpn_post_time)
assert len(head_time) == len(head_post_time)


print('backbone_time' , backbone_time.mean() , backbone_time.std())
print('neck_time' , neck_time.mean(), neck_time.std())
print('rpn_time' , rpn_time.mean(), rpn_time.std())
print('rpn_post_time' , rpn_post_time.mean(), rpn_post_time.std())
print('roi_time' , roi_time.mean(), roi_time.std())
print('head_time' , head_time.mean(), head_time.std())
print('head_post_time' , head_post_time.mean(), head_post_time.std())
