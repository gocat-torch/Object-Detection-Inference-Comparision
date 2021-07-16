python tools/test.py \
    configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712_cocolike_left.py  \
    work_dirs/faster_rcnn_r101_fpn_1x_voc0712_cocolike_left/epoch_1.pth \
    --eval bbox > infer_faster_rcnn_r101_fpn_epoch1.out
