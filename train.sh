
# python train_unet.py -data_dir data/crack_segmentation_dataset -model_dir model -model_type resnet101
#python train_unet.py -data_dir /home/ken/workspace/crack_segmentation/crack_image_old/inference/ -model_dir model -model_type resnet101 -n_epoch 20

python train_unet.py -data_dir /home/ken/workspace/crack_segmentation/crack_segmentation_dataset/train -model_dir model -model_type resnet101 -n_epoch 20