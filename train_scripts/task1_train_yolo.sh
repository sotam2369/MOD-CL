cd /usr/src/ROAD-R-2023-fork/
CUDA_VISIBLE_DEVICES=0 python main.py 1 /usr/src/road-dataset/ ./ ./ --MODE=train --YOLO --MAX_EPOCHS 19