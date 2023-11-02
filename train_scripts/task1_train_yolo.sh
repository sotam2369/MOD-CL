cd /usr/src/MOD-CL/
CUDA_VISIBLE_DEVICES=0 python main.py 1 /usr/src/road-dataset/ ./ ./ --MODE=train --YOLO --MAX_EPOCHS 19