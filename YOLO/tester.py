import types
import pickle
import os
import torch

from .model import ExtenderModel
from .postprocess import postprocess, track


def test(model, video_files, dataset_dir, tracking=None, save_output=False, out_name="result_output/final_results.pkl", extender=False):
    # Initiates the YOLO model, to modify the postprocessing methods
    model.predict('YOLO/test.jpg', save=False, save_txt=False, save_conf=False)
    model.predictor.postprocess = types.MethodType(postprocess, model.predictor)
    model.track = types.MethodType(track, model)

    if extender:
        ex_model = ExtenderModel()
        ex_model.load_state_dict(torch.load("YOLO/extender/extender_model.pt"))
        ex_model.eval()

    db_final = {}
    
    if not os.path.exists(os.path.dirname(out_name)):
        os.makedirs(os.path.dirname(out_name))

    for video_name in video_files:
        if tracking is None:
            test = model.predict(os.path.join(dataset_dir, video_name) + ".mp4", save=save_output, conf=0.05, device="cuda device=0", max_det=300, line_width=1, stream=True)
        else:
            test = model.track(os.path.join(dataset_dir, video_name) + ".mp4", save=save_output, conf=0.05, device="cuda device=0", tracker=tracking, max_det=300, line_width=1, stream=True)
        db = {}
        for res_id, res in enumerate(test):
            frame_name = "{:05}".format(int(res_id+1)) + ".jpg"
            frame_db= []
            for bbox_id in range(len(res.boxes)):
                bbox_db = {}
                bbox_db['bbox'] = res.boxes.xyxy[bbox_id].tolist()
                if extender:
                    with torch.no_grad():
                        bbox_db['labels'] = (res.boxes_all[bbox_id, 4:]*0.8 + ex_model(res.boxes_all[bbox_id, 4:]).sigmoid()*0.2).tolist()
                else:
                    bbox_db['labels'] = res.boxes_all[bbox_id, 4:].tolist()
                frame_db.append(bbox_db)
            db[frame_name] = frame_db
        db_final[video_name] = db

        with open(out_name, 'wb') as outfile:
            pickle.dump(db_final, outfile)