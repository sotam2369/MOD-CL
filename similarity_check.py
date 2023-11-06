import pickle
import sys
from tqdm import tqdm

if __name__ == '__main__':
    task = int(sys.argv[1])

    if task == 1:
        names = ['task1', 'final_output_task1']
    else:
        names = ['task2', 'final_validated_output']

    with open(f'submitted_output/{names[0]}.pkl', 'rb') as f:
        db_final = pickle.load(f)

    with open(f'result_output/{names[1]}.pkl', 'rb') as f:
        db_final_check = pickle.load(f)

    difference = 0

    for video in db_final.keys():
        for frame in tqdm(db_final[video].keys()):
            for bbox in db_final[video][frame]:
                if not bbox in db_final_check[video][frame]:
                    difference += 1
                else:
                    db_final_check[video][frame].remove(bbox)
            if len(db_final_check[video][frame]) == 0:
                db_final_check[video].pop(frame)

    if difference == 0:
        print("SAME!")
        print(db_final_check)
    else:
        print("Difference:", difference)