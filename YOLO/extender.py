import types
import os
import json
import pickle

import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader as TDLoader
import torch.nn.functional as F
import random

import numpy as np

from .model import ExtenderModel
from .postprocess import postprocess, track


# Function copied from the data/dataset.py file in ROAD-R-2023
def filter_labels(ids, all_labels, used_labels, offset):
    """Filter the used ids"""
    used_ids = []
    for id in ids:
        label = all_labels[id]
        if label in used_labels:
            used_ids.append(used_labels.index(label) + offset)
    
    return used_ids

# Setup function for preparing the dataset used in training the Extender Model
def setupExtenderModel(model, labelled_videos, unlabelled_videos, dataset_dir, tracking=None, thres_iou=0.75):
    print("Setting up for training the Extender Model")

    print("Generating training data...")
    
    # Makes the directory for the extender model
    os.makedirs('YOLO/extender')

    # Makes the initial predictions, and edits the postprocess functions (same as in the tester)
    model.predict('YOLO/test.jpg', save=False, save_txt=False, save_conf=False)
    model.predictor.postprocess = types.MethodType(postprocess, model.predictor)
    model.track = types.MethodType(track, model)


    with open(os.path.join(dataset_dir, "road/road_trainval_v1.0.json")) as f:
        ground_truth = json.load(f)


    gt_list = []
    input_list = []
    input_nogt_list = []

    # Loops through all the labelled videos
    for video_name in labelled_videos:
        # Gets the output from the YOLO model
        if tracking is None:
            test = model.predict(os.path.join(dataset_dir, "road/videos", video_name) + ".mp4", save=False, conf=0.05, device="cuda device=0", max_det=300, line_width=1, stream=True)
        else:
            test = model.track(os.path.join(dataset_dir, "road/videos", video_name) + ".mp4", save=False, conf=0.05, device="cuda device=0", tracker=tracking, max_det=300, line_width=1, stream=True)
        
        # Loops through all the frames in the video, with the YOLO outputs
        for res_id, res in enumerate(test):
            gt_bbox = []
            gt_labels = []
            if ground_truth['db'][video_name]['frames'][str(res_id+1)]['annotated'] == 1:
                gt_frame = ground_truth['db'][video_name]['frames'][str(res_id+1)]['annos']
                for bbox_name in gt_frame.keys():
                    gt_bbox.append(gt_frame[bbox_name]['box'])
                    id_labels = []
                    id_labels += filter_labels(gt_frame[bbox_name]['agent_ids'], ground_truth['all_agent_labels'], ground_truth['agent_labels'], 0) 
                    id_labels += filter_labels(gt_frame[bbox_name]['action_ids'], ground_truth['all_action_labels'], ground_truth['action_labels'], len(ground_truth['agent_labels']))
                    id_labels += filter_labels(gt_frame[bbox_name]['loc_ids'], ground_truth['all_loc_labels'], ground_truth['loc_labels'], len(ground_truth['agent_labels']) + len(ground_truth['action_labels']))
                    gt_label_now = torch.zeros(len(ground_truth['agent_labels']) + len(ground_truth['action_labels']) + len(ground_truth['loc_labels']), device=res.boxes.xyxy.device)
                    gt_label_now[id_labels] = 1
                    gt_labels.append(gt_label_now)
            
            # Loads the ground truth bounding boxes and labels
            gt_bbox = torch.tensor(gt_bbox, device=res.boxes.xyxy.device)
            if len(gt_labels) > 0:
                gt_labels = torch.stack(gt_labels, dim=0)
            
            # Loops through all the boxes predicted by YOLO
            for bbox_id in range(len(res.boxes)):
                # Appends the predictions to the input list
                input_list.append(res.boxes_all[bbox_id, 4:])
                if len(gt_bbox) == 0:
                    # If there are no ground truth boxes, a list of 0's will be appended to the gt_list
                    gt_list.append(torch.zeros((len(ground_truth['agent_labels']) + len(ground_truth['action_labels']) + len(ground_truth['loc_labels'])), device=res.boxes.xyxy.device))
                    continue
                bbox_now = torch.unsqueeze(res.boxes.xyxyn[bbox_id], dim=0)
                iou_now = torchvision.ops.box_iou(bbox_now, gt_bbox)
                iou_max = torch.max(iou_now, dim=1)
                
                if float(iou_max[0]) >= thres_iou:
                    # If there is a box that has an IoU of more than 0.75, the ground truth labels will be appended to the gt_list
                    gt_list.append(gt_labels[iou_max[1]][0])
                else:
                    # If there is no box that has an IoU of more than 0.75, a list of 0's will be appended to the gt_list
                    gt_list.append(torch.zeros((len(ground_truth['agent_labels']) + len(ground_truth['action_labels']) + len(ground_truth['loc_labels'])), device=res.boxes.xyxy.device))
    
    
    # Loops through all the unlabelled videos
    for video_name in unlabelled_videos:
        # Gets the output from the YOLO model
        if tracking is None:
            test = model.predict(os.path.join(dataset_dir, "road/videos", video_name) + ".mp4", save=False, conf=0.05, device="cuda device=0", max_det=300, line_width=1, stream=True)
        else:
            test = model.track(os.path.join(dataset_dir, "road/videos", video_name) + ".mp4", save=False, conf=0.05, device="cuda device=0", tracker=tracking, max_det=300, line_width=1, stream=True)
        
        # Loops through all the frames in the video, with the YOLO outputs
        for res_id, res in enumerate(test):
            # Loops through all the boxes predicted by YOLO
            for bbox_id in range(len(res.boxes)):
                # Appends the predictions to the input list (that has no ground truths)
                input_nogt_list.append(res.boxes_all[bbox_id, 4:])
    

    # Saves the generated dataset
    print("Number of training data: " + str(len(input_list)))
    with open('YOLO/extender/extender_input_list.pkl', 'wb') as outfile:
        pickle.dump(torch.stack(input_list), outfile)

    with open('YOLO/extender/extender_input_nogt_list.pkl', 'wb') as outfile:
        pickle.dump(torch.stack(input_nogt_list), outfile)
    
    with open('YOLO/extender/extender_gt_list.pkl', 'wb') as outfile:
        pickle.dump(torch.stack(gt_list), outfile)


def getTNormLoss(output, constraints, device):
    pred_const = output.sigmoid()
    pred_const = torch.cat([pred_const, 1.0-pred_const], axis=-1)
    loss_const = torch.ones((pred_const.shape[0], constraints.shape[0]), device=device)

    for req_id in range(constraints.shape[0]):
        loss_const[:,req_id] = torch.prod(1-pred_const[:,constraints.indices()[1][constraints.indices()[0]==req_id]], axis=-1)
    return loss_const.sum() / (loss_const.shape[0] * loss_const.shape[1])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




def startExtenderTraining(model, labelled_videos, unlabelled_videos, dataset_dir, tracking=None, thres_iou=0.75, epochs=5, seed=0):
    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    g = torch.Generator()
    g.manual_seed(seed)


    print("Starting training the Extender Model")
    # If the extender dataset does not exist, generate them
    if not os.path.isfile('YOLO/extender/extender_input_list.pkl'):
        setupExtenderModel(model, labelled_videos, unlabelled_videos, dataset_dir, tracking=tracking, thres_iou=thres_iou)

    with open('YOLO/extender/extender_input_list.pkl', 'rb') as f:
        input_list = pickle.load(f)

    with open('YOLO/extender/extender_input_nogt_list.pkl', 'rb') as f:
        input_nogt_list = pickle.load(f)
    
    with open('YOLO/extender/extender_gt_list.pkl', 'rb') as f:
        gt_list = pickle.load(f)


    # Load the dataset (with labels) into the dataloader
    labelled_dataset = TensorDataset(input_list, gt_list)
    ld_train, ld_test = torch.utils.data.random_split(labelled_dataset, [int(0.8*len(labelled_dataset)), len(labelled_dataset) - int(0.8*len(labelled_dataset))])

    ld_train_loader = TDLoader(ld_train, batch_size=256, shuffle=True, worker_init_fn=seed_worker, generator=g)
    ld_test_loader = TDLoader(ld_test, batch_size=256, shuffle=False, worker_init_fn=seed_worker, generator=g)
    dataset2_loader = TDLoader(TensorDataset(input_nogt_list), batch_size=256, shuffle=True, worker_init_fn=seed_worker, generator=g)

    # Load the Extender Model
    ex_model = ExtenderModel()
    optimizer = torch.optim.Adam(ex_model.parameters())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the Combiner Model
    combiner = torch.nn.Linear(2, 1)
    combiner_opt = torch.optim.Adam(combiner.parameters())

    # Load the constraints
    constraints_np = np.load("YOLO/constraints.npy")
    constraints = torch.from_numpy(constraints_np).to_sparse()

    
    ex_model.to(device)
    combiner.to(device)
    best = 100

    # Start the training
    for epoch in range(epochs):
        print("Epoch " + str(epoch))
        ex_model.train()
        combiner.train()

        # Stage 1: Train the Extender Model only, with unlabelled dataset, with T-Norm Loss
        loss_sum = torch.zeros((1), device=device)
        for batch_id, input in enumerate(dataset2_loader):
            optimizer.zero_grad()

            input = input[0].to(device)
            output = ex_model(input)

            loss = torch.zeros((1), device=device)
            loss[0] = getTNormLoss(output, constraints, device) # T-Norm Loss
            loss_sum = loss_sum + loss

            loss.sum().backward()
            optimizer.step()
            print(f"Stage 1 ({batch_id+1}):", loss_sum/(batch_id+1), end="\r", flush=True)

        print()

        # Stage 2: Train the Extender Model and the Combiner Model, with labelled dataset, with T-Norm Loss and Binary Cross Entropy Loss of the original labels
        loss_sum = torch.zeros((3), device=device)
        for batch_id, (input, gt) in enumerate(ld_train_loader):
            optimizer.zero_grad()

            input = input.to(device)
            gt = gt.to(device)

            output = ex_model(input)
            loss = torch.zeros((3), device=device)
            loss[0] = F.binary_cross_entropy_with_logits(output, gt, reduction='mean')
            loss[1] = getTNormLoss(output, constraints, device)*10

            output = combiner(torch.stack([output, input], dim=-1))
            loss[2] = F.binary_cross_entropy_with_logits(output, torch.unsqueeze(gt, dim=-1), reduction='mean')

            loss_sum = loss_sum + loss
            loss.sum().backward()
            optimizer.step()
            combiner_opt.step()
            print(f"Stage 2 ({batch_id+1}):", loss_sum/(batch_id+1), end="\r", flush=True)
        
        print()
        loss_sum = torch.zeros((3), device=device)
        ex_model.eval()
        combiner.eval()
        with torch.no_grad():
            for batch_id, (input, gt) in enumerate(ld_test_loader):
                input = input.to(device)
                gt = gt.to(device)

                output = ex_model(input)
                loss = torch.zeros((3), device=device)
                loss[0] = F.binary_cross_entropy_with_logits(output, gt, reduction='mean')
                loss[1] = getTNormLoss(output, constraints, device)*10

                output = combiner(torch.stack([output, input], dim=-1))
                loss[2] = F.binary_cross_entropy_with_logits(output, torch.unsqueeze(gt, dim=-1), reduction='mean')

                loss_sum = loss_sum + loss
                print(f"Validation ({batch_id+1}):", loss_sum/(batch_id+1), end="\r", flush=True)

        # Save the model with the best validation loss, in terms of the BCE Loss of the Combiner Model output
        if float(loss_sum[2]) < best:
            best = float(loss_sum[2])
            torch.save(ex_model.state_dict(), 'YOLO/extender/extender_model.pt')
            torch.save(combiner.state_dict(), 'YOLO/extender/combiner_model.pt')
            print("\n\nSAVED")
            
        print("\n\n")
        