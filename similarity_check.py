import pickle

names = ['corrected_best_new', 'corrected']
#names = ['final_results_best', 'final_results']

with open(f'result_output/{names[0]}.pkl', 'rb') as f:
    db_final = pickle.load(f)

with open(f'result_output/{names[1]}.pkl', 'rb') as f:
    db_final_check = pickle.load(f)


for video in db_final.keys():
    for frame in db_final[video].keys():
        for bbox in db_final[video][frame]:
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