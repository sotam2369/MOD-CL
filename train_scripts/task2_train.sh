cd /usr/src/MOD-CL/
CUDA_VISIBLE_DEVICES=0 python main.py 2 /usr/src/road-dataset/ /usr/src/experiments/ /usr/src/ROAD-R-2023-fork/kinetics-pt/ --MODE=gen_dets --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road --TRAIN_SUBSETS=train_1 --VAL_SUBSETS=test --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041 --NUM_WORKERS=8 --req_loss_weight=10.0 --LOGIC=Product 