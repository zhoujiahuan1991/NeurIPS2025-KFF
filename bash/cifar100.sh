export CUDA_VISIBLE_DEVICES=0
python main.py --cfg cfgs/cifar100.yaml cfgs/ours.yaml \
    --src_data_dir /data/dataset --data_dir /data/dataset/zhuchao/cifar/ \
    --checkpoint /home/zhuchao/pretrain_cifar100.t7 --train_info train_info_cifar100.pth