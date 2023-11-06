# Ultralytics YOLO 🚀, AGPL-3.0 license

from functools import partial

import torch

from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml

from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes

TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.

    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.
    """
    if hasattr(predictor, 'trackers') and persist:
        return
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))
    assert cfg.tracker_type in ['bytetrack', 'botsort'], \
        f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
    predictor.trackers = trackers


def on_predict_postprocess_end(predictor):
    """Postprocess detected boxes and update with object tracking."""
    bs = predictor.dataset.bs
    im0s = predictor.batch[1]
    for i in range(bs):
        det = predictor.results[i].boxes.cpu().numpy()
        if len(det) == 0:
            continue
        try:
            tracks = predictor.trackers[i].update(det, im0s[i])
        except:
            print("Detected error")
            continue
        if len(tracks) == 0:
            continue

        idx = tracks[:, -1].astype(int)
        boxes_all = predictor.results[i].boxes_all[idx]
        boxes_all_id = predictor.results[i].boxes_all_id.data

        for i2 in range(tracks.shape[0]):
            idx_single = idx[i2]
            
            equal = torch.argwhere(predictor.results[i].pred_index==idx_single).squeeze(1)
            boxes_all_id[equal, :4] = torch.as_tensor(tracks[i2, :4]).repeat(equal.shape[0], 1)
        
        predictor.results[i] = predictor.results[i][idx]
        boxes_all[:, :4] = torch.as_tensor(tracks[:, :4])

        predictor.results[i].boxes_all = boxes_all
        predictor.results[i].boxes_all_id = Boxes(boxes_all_id, predictor.results[i].orig_shape)
        predictor.results[i].tracks = tracks
        predictor.results[i].update(boxes=torch.as_tensor(tracks[:, :-1]))


def register_tracker(model, persist):
    """
    Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.

    """
    model.add_callback('on_predict_start', partial(on_predict_start, persist=persist))
    model.add_callback('on_predict_postprocess_end', on_predict_postprocess_end)