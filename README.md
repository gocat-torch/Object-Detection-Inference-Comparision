# Object-Detection-Inference-Comparision

GPU : TITAN RTX
Pytorch: 1.3

### Model: SSD

| Duration        | VGG-16           | MobileNetV2  ||
| ----------------|----------|------| ----|--|
| Backbone        | 0.00147 (± 0.00214)      | 0.00438  (± 0.00208) |
| Header          | 0.00105 (± 0.00008)      |  0.00170  (± 0.00003)|
| Box converting  | 0.00020 (± 0.00000)     |  0.00020  (± 0.00000)|
| NMS             | 0.02835 (± 0.00939)     |  0.01721  (± 0.00740)|




## Acknowledgements
For SSD, most of codes are borrowed from pytorch-ssd:
https://github.com/qfgaohao/pytorch-ssd
