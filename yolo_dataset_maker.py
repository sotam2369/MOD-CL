import json
import shutil
import os
from tqdm import tqdm
import random

# Function copied from the data/dataset.py file in ROAD-R-2023
def filter_labels(ids, all_labels, used_labels, offset):
    """Filter the used ids"""
    used_ids = []
    for id in ids:
        label = all_labels[id]
        if label in used_labels:
            used_ids.append(used_labels.index(label) + offset)
    
    return used_ids

def make_yolo_dataset(dataset_folder="../road-dataset/", task=1):
    # Seeding for the dataset generator
    random.seed(1)
    # Loads the json file including all the labels for the road-dataset
    with open(dataset_folder + "road/road_trainval_v1.0.json") as f:
        a = json.load(f)

    # Lists the videos that are labelled for the task
    if task == 1:
        # Three videos that can be used with labels for task 1
        labelled_videos = "2014-07-14-14-49-50_stereo_centre_01,2015-02-03-19-43-11_stereo_centre_04,2015-02-24-12-32-19_stereo_centre_04"
    elif task == 2:
        # All videos used for training in task 2
        # The remaining videos will be copied to the validation folder with labels
        labelled_videos = "2014-06-25-16-45-34_stereo_centre_02,2014-07-14-14-49-50_stereo_centre_01," \
                "2014-07-14-15-42-55_stereo_centre_03,2014-08-08-13-15-11_stereo_centre_01,2014-08-11-10-59-18_stereo_centre_02," \
                "2014-11-14-16-34-33_stereo_centre_06,2014-11-18-13-20-12_stereo_centre_05,2014-11-21-16-07-03_stereo_centre_01," \
                "2014-12-09-13-21-02_stereo_centre_01,2015-02-03-08-45-10_stereo_centre_02,2015-02-03-19-43-11_stereo_centre_04," \
                "2015-02-06-13-57-16_stereo_centre_02,2015-02-13-09-16-26_stereo_centre_05,2015-02-24-12-32-19_stereo_centre_04," \
                "2015-03-03-11-31-36_stereo_centre_01"

    labelled_videos = labelled_videos.split(",")
    train_id = 1
    val_id = 1

    # Checks if the dataset is already made
    if not os.path.isdir(f"{dataset_folder}yolo_road-r_task{task}/images/"):
        # If not, starts the generation of the dataset
        os.makedirs(f"{dataset_folder}yolo_road-r_task{task}/images/train")
        os.makedirs(f"{dataset_folder}yolo_road-r_task{task}/images/val")
        os.makedirs(f"{dataset_folder}yolo_road-r_task{task}/labels/train")
        os.makedirs(f"{dataset_folder}yolo_road-r_task{task}/labels/val")

        for label in a['db'].keys():
            if label in labelled_videos:
                # As the best epochs were already found using cross validation, we can use the whole dataset for training. (As YOLO requires a validation dataset, we copy some of the training images to the validation folder)
                for frame in tqdm(a['db'][label]['frames']):
                    frame_name = "{:05}".format(int(frame))
                    train_id_str = "{:05}".format(train_id)

                    # Adds the images to the YOLO dataset (training)
                    shutil.copyfile(dataset_folder + "road/rgb-images/" + label + "/" + frame_name + ".jpg", f"{dataset_folder}yolo_road-r_task{task}/images/train/" + train_id_str + ".jpg")

                    # As the validation and training labels for the same image are configured differently in this model, we make two versions of the same labels
                    total_output = []
                    total_output2 = []
                    if a['db'][label]['frames'][frame]['annotated'] == 1:
                        for box_name in a['db'][label]['frames'][frame]['annos'].keys():
                            id_labels = []
                            id_labels += filter_labels(a['db'][label]['frames'][frame]['annos'][box_name]['agent_ids'], a['all_agent_labels'], a['agent_labels'], 0) 
                            id_labels += filter_labels(a['db'][label]['frames'][frame]['annos'][box_name]['action_ids'], a['all_action_labels'], a['action_labels'], len(a['agent_labels']))
                            id_labels += filter_labels(a['db'][label]['frames'][frame]['annos'][box_name]['loc_ids'], a['all_loc_labels'], a['loc_labels'], len(a['agent_labels']) + len(a['action_labels']))

                            box = a['db'][label]['frames'][frame]['annos'][box_name]['box']
                            box = [(box[0] + box[2])/2,(box[1] + box[3])/2,box[2]-box[0],box[3]-box[1]]
                            for i in range(len(box)):
                                box[i] = min(1, max(0, box[i]))
                            box = list(map(str, box))
                            box_string = ' '.join(box)
                            id_labels = list(map(str, id_labels))
                            total_output.append(','.join(id_labels) + " " + box_string + "\n")
                            for id_label in id_labels:
                                total_output2.append(str(id_label) + " " + box_string + "\n")
                    
                    with open(f"{dataset_folder}yolo_road-r_task{task}/labels/train/" + train_id_str + ".txt", "w") as f:
                        f.writelines(total_output)
                    
                    # If the random value is low enough and the task is 1, the image is saved to the validation folder, as well as the training folder.
                    if task == 1 and random.random() <= 0.3:
                        shutil.copyfile(dataset_folder + "road/rgb-images/" + label + "/" + frame_name + ".jpg", f"{dataset_folder}yolo_road-r_task{task}/images/val/" + train_id_str + ".jpg")
                        with open(f"{dataset_folder}yolo_road-r_task{task}/labels/val/" + train_id_str + ".txt", "w") as f:
                            f.writelines(total_output2)

                    train_id += 1
            elif task != 1:
                # For task 2, the videos that are not in the labelled list are copied to the validation folder with labels
                for frame in tqdm(a['db'][label]['frames']):
                    frame_name = "{:05}".format(int(frame))
                    val_id_str = "{:05}".format(val_id)

                    # Adds the images to the YOLO dataset (validating)
                    shutil.copyfile(dataset_folder + "road/rgb-images/" + label + "/" +  frame_name + ".jpg", f"{dataset_folder}yolo_road-r_task{task}/images/val/" + val_id_str + ".jpg")
                    total_output = []
                    if a['db'][label]['frames'][frame]['annotated'] == 1:
                        for box_name in a['db'][label]['frames'][frame]['annos'].keys():
                            id_labels = []
                            id_labels += filter_labels(a['db'][label]['frames'][frame]['annos'][box_name]['agent_ids'], a['all_agent_labels'], a['agent_labels'], 0) 
                            id_labels += filter_labels(a['db'][label]['frames'][frame]['annos'][box_name]['action_ids'], a['all_action_labels'], a['action_labels'], len(a['agent_labels']))
                            id_labels += filter_labels(a['db'][label]['frames'][frame]['annos'][box_name]['loc_ids'], a['all_loc_labels'], a['loc_labels'], len(a['agent_labels']) + len(a['action_labels']))

                            box = a['db'][label]['frames'][frame]['annos'][box_name]['box']
                            box = [(box[0] + box[2])/2,(box[1] + box[3])/2,box[2]-box[0],box[3]-box[1]]
                            for i in range(len(box)):
                                box[i] = min(1, max(0, box[i]))
                            box = list(map(str, box))
                            box_string = ' '.join(box)
                            for id_label in id_labels:
                                total_output.append(str(id_label) + " " + box_string + "\n")
                    
                    with open(f"{dataset_folder}yolo_road-r_task{task}/labels/val/" + val_id_str + ".txt", "w") as f:
                        f.writelines(total_output)
                    val_id += 1
    else:
        print("Dataset is already made!")

if __name__ == "__main__":
    make_yolo_dataset(task=1)