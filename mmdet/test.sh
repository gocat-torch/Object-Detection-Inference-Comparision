python tools/test.py \
    configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712_cocolike.py  \
    work_dirs/faster_rcnn_r101_fpn_1x_voc0712_cocolike/epoch_1.pth \
    --eval bbox > infer_faster_rcnn_r101_fpn_epoch1.out
