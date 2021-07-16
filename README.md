7# Object-Detection-Inference-Comparision

GPU : TITAN RTX
Pytorch: 1.3

### Model: SSD

| Duration        | : VGG-16          : || : MobileNetV2    : || 
|-----------------|---------|---------|-------|--------|
| Training        | before  | after  | before  | after |
| Backbone        |0.00147 (± 0.00217) | 0.00147 (± 0.00214)      | 0.00440 (± 0.00207) | 0.00438  (± 0.00208) |
| Header          |0.00104 (± 0.00010) | 0.00105 (± 0.00008)      | 0.00170 (± 0.00003)|  0.00170  (± 0.00003) |
| Box converting  |0.00020 (± 0.00000) | 0.00020 (± 0.00000)      | 0.00020 (± 0.00000)|  0.00020  (± 0.00000) |
| NMS             |0.13452 (± 0.00522) | 0.02835 (± 0.00939)      | 0.08391 (± 0.00533)|  0.01721  (± 0.00740)|

### Model : Faster-RCNN ( ResNet101-FPN)
| Duration                 | : ResNet101-FPN          : ||  || 
|-----------------         |---------|--------- |-------|--------|
| Training                 | before  | after    | before  | after |
| Backbone                 |0.00882 (± 0.00250) | 0.00852 (± 0.00248)      | |  |
| Neck                     |0.00057 (± 0.00002) | 0.00056 (± 0.00002)      | |  |
| RPN                      |0.00070 (± 0.00003) | 0.00067 (± 0.00003)      | |  |
| Post RPN (Including NMS) |0.02766 (± 0.00300) | 0.02783 (± 0.00299)      | |  |
| ROI Pooling              |0.00279 (± 0.00014) | 0.00274 (± 0.00014)      | |  |
| Head                     |0.00019 (± 0.00000) | 0.00018 (± 0.00000)      | |  |
| Post Head (Including NMS)|0.00406 (± 0.00024) | 0.00403 (± 0.00025)      | |  |


## Acknowledgements
For SSD, most of codes are borrowed from pytorch-ssd:
https://github.com/qfgaohao/pytorch-ssd
