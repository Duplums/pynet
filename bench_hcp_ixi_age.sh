main=~/PycharmProjects/pynet/main.py

CUDA_VISIBLE_DEVICES=1 python3 $main --checkpoint_dir /neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/tmp --exp_name ResNet24_DX -b 8 --net resnet34 --lr 1e-4 --nb_epochs 100 --test --pretrained_path /neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/Benchmark_IXI_HCP/ResNet34_IXI_HCP_0_epoch_67.pth &> resnet34.txt &
#CUDA_VISIBLE_DEVICES=1 python3 $main --checkpoint_dir /neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/tmp --exp_name ResNet101_HCP_IXI -b 16 --net resnet101 --lr 1e-4 --nb_epochs 100 &> resnet101.txt &
#CUDA_VISIBLE_DEVICES=2 python3 $main --checkpoint_dir /neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/tmp --exp_name VGG11_HCP_IXI -b 16 --net vgg11 --lr 1e-4 --nb_epochs 100 &> vgg11.txt &
#CUDA_VISIBLE_DEVICES=3 python3 $main --checkpoint_dir /neurospin/psy_sbox/bd261576/checkpoints/regression_age_sex/tmp --exp_name VGG16_HCP_IXI -b 16 --net vgg16 --lr 1e-4 --nb_epochs 100 &> vgg16.txt &