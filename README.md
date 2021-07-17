# Object-Detection-Inference-Comparision

GPU : TITAN RTX
Pytorch: 1.3

### Model: SSD (Dataset : Pascal VOC)

| Duration        | VGG-16   || MobileNetV2    || 
|-----------------|---------|---------|-------|--------|
| Training        | before  | after  | before  | after |
| Backbone        |0.00147 (± 0.00217) | 0.00147 (± 0.00214)  | 0.00440 (± 0.00207) | 0.00438  (± 0.00208) |
| Header          |0.00104 (± 0.00010) | 0.00105 (± 0.00008)  | 0.00170 (± 0.00003)|  0.00170  (± 0.00003) |
| Box converting  |0.00020 (± 0.00000) | 0.00020 (± 0.00000) | 0.00020 (± 0.00000)|  0.00020  (± 0.00000) |
| NMS             |0.13452 (± 0.00522) | 0.02835 (± 0.00939)   | 0.08391 (± 0.00533)|  0.01721  (± 0.00740)|

### Model : Faster-RCNN ( ResNet101-FPN) ( Dataset : COCO)
| Duration                 | ResNet50-FPN ||  ResNet101-FPN           || 
|-----------------         |---------|--------- |-------|--------|
| Training                 | before  | after    | before             | after                    |
| Backbone                 |         | 0.00466 (± 0.00249)         | | 0.00869 (± 0.00255)      |
| Neck                     |         | 0.00073 (± 0.00014)         | | 0.00061 (± 0.00009)      |
| RPN                      |         | 0.00077 (± 0.00005)         | | 0.00073 (± 0.00005)      |
| Post RPN (Including NMS) |         | 0.03655 (± 0.00354)         | | 0.04905 (± 0.00510)      |
| ROI Pooling              |         | 0.00292 (± 0.00014)         | | 0.00287 (± 0.00013)      |
| Head                     |         | 0.00019 (± 0.00000)         | | 0.00019 (± 0.00000)      |
| Post Head (Including NMS)|         | 0.00401 (± 0.00014)         | | 0.00422 (± 0.00025)      |


### Model : CenterNetV2 (ResNet50) ( Dataset : COCO)

| Duration                 |  ResNet50           ||  | DLS_BiFPN-P3 | 
|-----------------         |---------|--------- |-------|--------|
| Training                 | before  | after    | before  | after |
| Backbone                 |  | 0.00524 (± 0.00331)      | | 0.00835 (± 0.00272)|
| Centernet Head           |  | 0.00285 (± 0.00003)      | | 0.00166 (± 0.00002) |
| Comptuing Grids          |  | 0.02441 (± 0.00275)      | | 0.00086 (± 0.00033) |
| Generationg Proposals    |  | 0.00318 (± 0.00158)      | | 0.00192 (± 0.00122) |
| Proposal NMS             |  | 0.00125 (± 0.00004)      | | 0.00097 (± 0.00142) |
| ROI Heads Cascade        |  | 0.00627 (± 0.00032)      | | 0.00442 (± 0.00009) |
| ROI Heads predict        |  | 0.00058 (± 0.00001)      | | 0.00056 (± 0.00001) |
| ROI Heads NMS            |  | 0.00086 (± 0.00001)      | | 0.00082 (± 0.00001) |
| Post Processing          |  | 0.00051 (± 0.00000)      | | 0.00050 (± 0.00000) |




## Acknowledgements
For SSD, most of codes are borrowed from pytorch-ssd:
https://github.com/qfgaohao/pytorch-ssd

For Faster R-CNN, most of codes are borrowed from mmdet:
https://github.com/open-mmlab/mmdetection
