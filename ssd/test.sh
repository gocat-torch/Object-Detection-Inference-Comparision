


CUDA_VISIBLE_DEVICES=0 python eval_ssd.py --net mb2-ssd-lite  --dataset /data/VOC/VOC_test/test --trained_model models_scratch/mb2-ssd-lite-Epoch-199-Loss-2.908908865528722.pth --label_file models/voc-model-labels.txt > mb2_result.out

CUDA_VISIBLE_DEVICES=0 python eval_ssd.py --net vgg16-ssd  --dataset /data/VOC/VOC_test/test --trained_model models_scratch/vgg16-ssd-Epoch-199-Loss-3.318957429001297.pth --label_file models/voc-model-labels.txt > vgg_result.out

