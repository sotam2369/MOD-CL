cd /usr/src/MOD-CL/
CUDA_VISIBLE_DEVICES=0 python main.py 2 /usr/src/road-dataset/ ./ ./ --MODE=train --YOLO --LOGIC True --MAX_EPOCHS 100