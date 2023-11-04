import pickle
import sys


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

    same_digits = 10
    same_digits2 = 10
    for video in db_final_check.keys():
        for frame in db_final_check[video].keys():
            for bbox_num, bbox in enumerate(db_final_check[video][frame]):
                for i in range(4):
                    db_final_check[video][frame][bbox_num]['bbox'][i] = round(bbox['bbox'][i], same_digits)
                for i in range(len(bbox['labels'])):
                    db_final_check[video][frame][bbox_num]['labels'][i] = round(bbox['labels'][i], same_digits2)

    for video in db_final.keys():
        for frame in db_final[video].keys():
            for bbox in db_final[video][frame]:
                for i in range(4):
                    bbox['bbox'][i] = round(bbox['bbox'][i], same_digits)
                for i in range(len(bbox['labels'])):
                    bbox['labels'][i] = round(bbox['labels'][i], same_digits2)
                if not bbox in db_final_check[video][frame]:
                    print(bbox)
                    print(db_final_check[video][frame])
                    exit()
                else:
                    db_final_check[video][frame].remove(bbox)
            if len(db_final_check[video][frame]) == 0:
                db_final_check[video].pop(frame)

    print("SAME!")
    print(db_final_check)