import numpy as np
import torch
import pickle
from tqdm import tqdm

def checkOutput(const, labels):
    label_nhot = torch.zeros((41))
    label_nhot[labels] = 1
    label_nhot = torch.cat((label_nhot, 1-label_nhot)).unsqueeze(1).float()
    #print(const[torch.squeeze(torch.matmul(const, label_nhot) == 0)])
    return int(torch.sum(torch.matmul(const, label_nhot) == 0)) == 0

if __name__ == "__main__":
    print("Starting check of output")
    constraints_np = np.load("YOLO/constraints.npy")
    constraints = torch.from_numpy(constraints_np)[:243].float() # 1, 1, num_req, num_labels

    with open("result_output/corrected.pkl", "rb") as f:
        output = pickle.load(f)
    
    for video in output.keys():
        i = 0
        total = 0
        for frame in tqdm(output[video].keys()):
            for bbox in output[video][frame]:
                if not checkOutput(constraints, bbox["labels"]):
                    i += 1
                total += 1
        print("Number of errors in video " + video + ": " + str(i))
        print("Total number of bboxes in video " + video + ": " + str(total))
