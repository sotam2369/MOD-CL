cd /usr/src/ROAD-R-2023-fork/
python main.py 2 /usr/src/road-dataset/ ./ ./ --MODE=train --YOLO --MODE eval_frames --YOLO_PRETRAINED $1
python post_processing.py --post_proc map_times_pred_based --th 0.3
cd ./result_output/
zip corrected.zip corrected.pkl