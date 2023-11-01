from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.plotting import plot_labels
from ultralytics.utils import DEFAULT_CFG, RANK, colorstr, LOGGER, TQDM
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.tal import dist2bbox, make_anchors

from copy import copy
import numpy as np
from .loss import YOLOLoss
from .dataset import ROADYOLODataset
from .validator import YOLOValidator
import torch
from torch import distributed as dist
import time
import warnings
import types


class YOLOTrainer(DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, req_loss=False):
        super().__init__(cfg, overrides, _callbacks)
        self.req_loss = req_loss
    
    def build_dataset(self, img_path, mode='train', batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs, lb_class_id=self.model.lb_class_id, lb_id_class=self.model.lb_id_class, lb_id_class_norm=self.model.lb_id_class_norm)
        self.model.lb_class_id = dataset.lb_class_id
        self.model.lb_id_class = dataset.lb_id_class
        self.model.lb_id_class_norm = dataset.lb_id_class_norm
        return dataset

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = []
        for lb in self.train_loader.dataset.labels:
            for lb_single in lb['cls']:
                cls.append(self.train_loader.dataset.lb_id_class[lb_single[0]])
        cls = np.array(cls, dtype=object)
        plot_labels(boxes, cls, names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = YOLOModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1, req_loss=self.req_loss)
        model.lb_class_id = {}
        model.lb_id_class = {}
        model.lb_id_class_norm = {}
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        if self.req_loss:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'req_loss'
        else:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        validator = YOLOValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
        return validator



class YOLOModel(DetectionModel):

    def __init__(self, cfg='yolov8x.yaml', ch=3, nc=None, verbose=True, req_loss=False):
        super().__init__(cfg, ch, nc, verbose)
        self.req_loss = req_loss

    def init_criterion(self):
        crit = YOLOLoss(self, req_loss=self.req_loss)
        return crit



def build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32, lb_class_id={}, lb_id_class={}, lb_id_class_norm={}):
    """Build YOLO Dataset"""
    return ROADYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0,
        lb_class_id=lb_class_id,
        lb_id_class=lb_id_class,
        lb_id_class_norm=lb_id_class_norm)


"""

def forward(self, x):
    shape = x[0].shape  # BCHW
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x
    elif self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape

    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
        box = x_cat[:, :self.reg_max * 4]
        cls = x_cat[:, self.reg_max * 4:]
    else:
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

    if self.export and self.format in ('tflite', 'edgetpu'):
        # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
        # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
        # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
        img_h = shape[2] * self.stride[0]
        img_w = shape[3] * self.stride[0]
        img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
        dbox /= img_size
    prev_dtype = cls.dtype
    cls = self.ms_model(cls.sigmoid().transpose(-1,-2)).to(dtype=prev_dtype).transpose(-1,-2)*0.1 + cls
    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y if self.export else (y, x)
"""