cd ..
rm -rf YOLO_task1

cd train_scripts/

bash task1_train_yolo.sh
bash task1_train_extender.sh 0
bash task1_postprocess.sh 0