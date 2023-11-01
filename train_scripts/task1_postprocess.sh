cd /usr/src/ROAD-R-2023-fork/
python main.py 1 /usr/src/road-dataset/ ./ ./ --MODE=train --YOLO --MODE eval_frames --YOLO_PRETRAINED $1
cd ./result_output/
zip final_results.zip final_results.pkl