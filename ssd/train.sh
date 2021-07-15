

CUDA_VISIBLE_DEVICES=1 python train_ssd.py --datasets /data/VOC/VOCdevkit/VOC2007 /data/VOC/VOCdevkit/VOC2012 --validation_dataset /data/VOC/VOC_test/test --net vgg16-ssd --base_net models/vgg16_reducedfc.pth  --batch_size 24 --num_epochs 600 --scheduler multi-step --milestones  "360,480" --checkpoint_folder ./models --validation_epochs 30 

