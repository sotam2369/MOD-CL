from ultralytics.utils import ops, LOGGER
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.engine.results import Results, Boxes, Masks
from ultralytics.data.augment import LetterBox



import torch
from copy import deepcopy
import torchvision
import time
from .yolo_tracker import register_tracker

def postprocess(possible_self, preds, img, orig_imgs):
    """Post-processes predictions and returns a list of Results objects."""

    preds, preds_def, preds_def_all, pred_index = non_max_suppression_v2(preds,
                                    possible_self.args.conf,
                                    possible_self.args.iou,
                                    agnostic=possible_self.args.agnostic_nms,
                                    max_det=possible_self.args.max_det,
                                    classes=possible_self.args.classes,
                                    multi_label=True)

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i]
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        preds_def_all[i][:, :4] = ops.scale_boxes(img.shape[2:], preds_def_all[i][:, :4], orig_img.shape)
        preds_def[i][:, :4] = ops.scale_boxes(img.shape[2:], preds_def[i][:, :4], orig_img.shape)

        img_path = possible_self.batch[0][i]
        res = CustomResults(orig_img, path=img_path, names=possible_self.model.names, boxes=pred)

        res.boxes_all = preds_def[i].detach().to('cpu')
        res.boxes_all_id = Boxes(preds_def_all[i].detach().to('cpu'), res.orig_shape)
        res.pred_index = pred_index[i].detach().to('cpu')
        #print(res.boxes, res.boxes_all_id)
        results.append(res)
    #print(results[0].boxes_all)
    return results


def track(possible_self, source=None, stream=False, persist=False, **kwargs):
    """
    Perform object tracking on the input source using the registered trackers.

    Args:
        source (str, optional): The input source for object tracking. Can be a file path or a video stream.
        stream (bool, optional): Whether the input source is a video stream. Defaults to False.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
        **kwargs (optional): Additional keyword arguments for the tracking process.

    Returns:
        (List[ultralytics.engine.results.Results]): The tracking results.
    """
    if not hasattr(possible_self.predictor, 'trackers'):
        register_tracker(possible_self, persist)
    # ByteTrack-based method needs low confidence predictions as input
    kwargs['conf'] = kwargs.get('conf') or 0.1
    kwargs['mode'] = 'track'
    return possible_self.predict(source=source, stream=stream, **kwargs)


def non_max_suppression_v2(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
        single_agents=True
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    xc = prediction[:, 4:14].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = ops.xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    output_def = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    output_all = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    output_index = [torch.zeros((0,), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence for all classes
        x_unedited = x.clone()  # for later comparison


        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = ops.xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)


        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        #print(x.shape)
        if multi_label and not single_agents:
            i, j = torch.where(cls > conf_thres) # Filter out non related classes
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls[:,:10].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes


        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        #print(i)
        #print(x)

        x_unedited = x_unedited[i]
        x = x[i]
        #print(x_unedited.shape)
        #print(x[i])

        box, cls, mask = x_unedited.split((4, nc, nm), 1)
        #print(box,cls,mask)
        if single_agents:
            i, j = torch.where(cls > conf_thres) # Filter out non related classes
            #print(len(i), i)
            x_all = torch.cat((box[i], x_unedited[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            #print(x.shape)

        

        # # Experimental
        # merge = False  # use merge-NMS
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     redundant = True  # require redundant detections
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy
        #print(torch.unique(x[:, :4], dim=0).shape)

        output[xi] = x
        output_def[xi] = x_unedited
        output_all[xi] = x_all
        output_index[xi] = i
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output, output_def, output_all, output_index


class CustomResults(Results):

    def update(self, boxes=None, masks=None, probs=None):
        """Update the boxes, masks, and probs attributes of the Results object."""
        if boxes is not None:
            ops.clip_boxes(boxes, self.orig_shape)  # clip boxes
            self.boxes = Boxes(boxes, self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = probs

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font='Arial.ttf',
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
    ):
        """
        Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.

        Args:
            conf (bool): Whether to plot the detection confidence score.
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            img (numpy.ndarray): Plot to another image. if not, plot to original image.
            im_gpu (torch.Tensor): Normalized image in gpu with shape (1, 3, 640, 640), for faster mask plotting.
            kpt_radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            masks (bool): Whether to plot the masks.
            probs (bool): Whether to plot classification probability

        Returns:
            (numpy.ndarray): A numpy array of the annotated image.

        Example:
            ```python
            from PIL import Image
            from ultralytics import YOLO

            model = YOLO('yolov8n.pt')
            results = model('bus.jpg')  # results list
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                im.show()  # show image
                im.save('results.jpg')  # save image
            ```
        """
        #print("\n\n\n\n\n\n")
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        pred_boxes, show_boxes = self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names)

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device).permute(
                    2, 0, 1).flip(0).contiguous() / 255
            idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect results
        if pred_boxes and show_boxes:
            plot_xyxy_unique = pred_boxes.xyxy.detach().to('cpu')
            plot_xyxy = self.boxes_all_id.xyxy
            plot_cls = self.boxes_all_id.cls
            plot_conf = self.boxes_all_id.conf
            plot_id = pred_boxes.id
            temp = 0
            for xyxy in plot_xyxy_unique:
                same_index = torch.sum(plot_xyxy==xyxy, dim=1) > 0
                c = torch.sort(plot_cls[same_index])[0]
                
                conf_now = plot_conf[same_index]
                final_label = []
                for i in range(c.shape[0]):
                    name = names[int(c[i])]
                    #final_label += [str(int(c[i]))]
                    final_label += [(f'{name} {conf_now[i]:.2f}' if not conf else name) if labels else None]
                #print(plot_id[0])
                if final_label == []:
                    #print(self.boxes_all_id, pred_boxes, self.boxes_all)
                    exit()
                final_label[0] = ('' if plot_id is None else f'id:{plot_id[temp].item()} ') + final_label[0]
                c_agent = c[c < 11]
                conf_agent = conf_now[c < 11]
                annotator.box_label(xyxy, ','.join(final_label), color=colors(int(c_agent[torch.argmax(conf_agent)]), True))
                temp += 1

            
            """
            for d in reversed(pred_boxes):
                print(d.xyxy, d.cls, d.conf, d.id)
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ('' if id is None else f'id:{id} ') + names[c]
                label = (f'{name} {conf:.2f}' if conf else name) if labels else None
                annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
                print(d.xyxy.squeeze())
                exit()"""

        #print(plot_xyxy[0], plot_cls[0], plot_conf[0], torch.sum(plot_xyxy==plot_xyxy_unique[0], dim=1) > 0)
        #exit()
        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = ',\n'.join(f'{names[j] if names else j} {pred_probs.data[j]:.2f}' for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        # Plot Pose results
        if self.keypoints is not None:
            for k in reversed(self.keypoints.data):
                annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

        return annotator.result()
    
    def new(self):
        """Return a new Results object with the same image, path, and names."""
        return CustomResults(orig_img=self.orig_img, path=self.path, names=self.names)