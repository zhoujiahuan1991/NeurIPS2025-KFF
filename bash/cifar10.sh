export CUDA_VISIBLE_DEVICES=0
python main.py --cfg cfgs/cifar10.yaml cfgs/ours.yaml \
    --src_data_dir /data/dataset --data_dir /data/dataset/zhuchao/cifar/ \
    --checkpoint /home/zhuchao/vit_base_384_cifar10.t7 --train_info train_info_cifar10.pth