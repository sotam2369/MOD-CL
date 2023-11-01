# MOD-CL: Multi-label Object Detection with Constraint Loss

Made by Team MWIT for the [ROAD-R Competition (2023)](https://sites.google.com/view/road-r/home?authuser=0).
<!-- 
Members include

- Sota Moriyama
- Akihiro Takemura
- Koji Watanabe
- Katsumi Inoue

All belong to the National Institute of Informatics, Japan.
!-->

## Specifications

### Model Architecture Specifications
The models used in both tasks are built upon [YOLOv8](https://github.com/ultralytics/ultralytics), a State of the Art Object Detection Model released by Ultralytics.
The pretrained weights used in this model are linked below.
- [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt)
- [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) (Mostly unused, only for loading purposes)

As the original YOLOv8 only supports single labels per bounding box, we have modified the program to **support multiple labels**. We used n-hot vectors as ground truths instead of the original one-hot vectors, to allow the bounding boxes to have high confidence scores for multiple labels at a time. Additionally, we focused on using a non-maximum suppression algorithm specifically tuned for the ROAD dataset. The modifications we made were mainly the following:
- Only uses candidate boxes which has confidence scores above the threshold for **any agent**
- Cut excess bounding boxes with respect to the confidence scores of the agent labels



### Task 1
We trained the YOLOv8 model on the given videos for 19 epochs, which was the optimal value found with cross-validation. Additionally, to exploit the unlabelled parts of the dataset, we introduced new models called the Extender Model and Combiner Model. We trained these models on the labelled parts of the dataset, as well as the unlabelled parts of the dataset.

### Task 2
We trained the YOLOv8 model with added constraint loss.
- Constraint Loss based on Product T-Norm
- Calculated the constraint loss on bounding boxes that have at least one label with a confidence score above 0.5
- Weighted the constraint loss by 10

## Reproducing test results

### Loading the models

Go to root folder. Then, run the following commands.
```
# Go to the parent folder, and git clone the ROAD dataset if not done so already
cd ..
git clone https://github.com/gurkirt/road-dataset.git
cd MOD-CL

# Make the docker image
sudo docker build -f scripts/Dockerfile -t MOD-CL .

# Run the docker
sudo docker run -it --ipc=host --gpus all --name MOD-CL-Cont -v "$(pwd)"/../road-dataset:/usr/src/road-dataset MOD-CL

# Run the installation scripts

bash scripts/installation_nodataset.sh
```
