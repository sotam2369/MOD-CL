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
We trained the YOLOv8 model on the given videos for 19 epochs, which was the optimal value found through cross-validation. Additionally, to exploit the unlabelled parts of the dataset, we introduced new models called the Extender Model and Combiner Model. We trained these models on the labelled parts of the dataset, as well as the unlabelled parts of the dataset. The specifics of these models are as follows:

Extender Model
- Takes the confidence scores output by YOLO as the input
- Outputs new confidence scores that more satisfies the constraints (requirements)
- Uses the sum of Binary Cross Entropy Loss and constraint loss (based on product T-Norm) as the loss function for labelled videos
- Uses constraint loss only for unlabelled videos


Combiner Model
- Takes the confidence scores output by YOLO and the Extender Model as input
- Outputs new confidence scores that combines the two scores
- Trains only on the labelled parts of the dataset
- Uses Binary Cross Entropy Loss as the loss function


### Task 2
We trained the YOLOv8 model with added constraint loss. We used constraint loss built on Product T-Norm, a method shown in the original ROAD-R paper (the code can be seen in lines 101-115 in [YOLO/loss.py](YOLO/loss.py)). In our model, we calculated the loss with the following steps:

1. Focused on bounding boxes which has at least one label with a confidence score above 0.5
1. Calculated how much each of the 243 requirements are **satisfied** using Product T-Norm (by transforming the conjunctions to disjunctions)
1. Calculated the average
1. Given it a weight of 10 when adding to other losses

## Reproducing test results

### Loading the enviroment

Go to root folder. Then, run the following commands.
```
# Go to the parent folder, and git clone the ROAD dataset if not done so already
cd ..
git clone https://github.com/gurkirt/road-dataset.git
cd MOD-CL

# Make the docker image
sudo docker build -f scripts/Dockerfile -t mod-cl .

# Run the docker
sudo docker run -it --ipc=host --gpus all --name mod-cl-cont -v "$(pwd)"/../road-dataset:/usr/src/road-dataset mod-cl

# Run the installation scripts

bash scripts/installation_nodataset.sh
```


### Running the training

Go to the scripts folder. Then, run the following commands.
```
# For training Task 1
bash task1.sh


# For training Task 2
bash task2.sh
```

The results of the execution will be saved in the result_output folder, which will be generated automatically.
- For Task 1, the final output will be result_output/final_results.pkl
- For Task 2, the final output will be result_output/final_validated_output.pkl