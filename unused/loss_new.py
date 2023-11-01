from ultralytics.utils.loss import *
from ultralytics.utils.tal import *

import torch
import torch.nn as nn

import numpy as np
#from pysat.formula import WCNFPlus
#from torch_geometric.data import Data, Batch


class YOLOLoss(v8DetectionLoss):
    def __init__(self, model, req_loss=False):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = CustomTaskAlignedAssigner(topk=20, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.assigner.lb_class_id = model.lb_class_id
        self.assigner.lb_id_class = model.lb_id_class
        self.assigner.lb_id_class_norm = model.lb_id_class_norm
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.req_loss = req_loss
        if req_loss:
            self.clause_init = None



    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, targets.shape[1]-1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), targets.shape[1]-1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""

        loss = torch.zeros(3 + int(self.req_loss), device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()


        if self.req_loss:
            if self.clause_init is None:
                #wcnf = WCNFPlus(from_file='constraints/WDIMACSconstraints_task2.txt')
                self.ms_model.to(self.device)
                self.ms_model.eval()
                #self.clause_init = torch.tensor([0 for _ in range(len(wcnf.hard))], device=self.device).float()
                #self.edge_attr, self.edge_index, self.mask = toGraph(wcnf)
                #self.edge_attr = torch.tensor(self.edge_attr, device=self.device).float()
                #self.edge_index = torch.tensor(self.edge_index, device=self.device).long()
                #self.mask = torch.tensor(self.mask, device=self.device).long()

                #x_feat_tensor = torch.unsqueeze(torch.cat((torch.tensor(data[0]),clause_init)), dim=-1)
                #self.constraints = torch.from_numpy(constraints_np).to_sparse() # 1, 1, num_req, num_labels
                #self.constraints_minus = torch.from_numpy((1-constraints_np)).to_sparse()#.to(self.device).to(torch.float16)
            #res = []
            prev_dtype = pred_scores.dtype
            pred_sig = pred_scores.sigmoid()
            #print(pred_scores)
            pred_scores = self.ms_model(pred_sig.float()).to(dtype=prev_dtype)*0.1 + pred_scores
            #print(pred_scores)
            """
            thres = 0.5
            pred_sig = pred_scores.sigmoid()
            pred_sig_obj = torch.max(pred_sig[:10], dim=-1)
            """"""
            for pred_now in pred_sig[pred_sig_obj[0] > thres]:
                x_feat = torch.unsqueeze(torch.cat((pred_now, self.clause_init), dim=-1), dim=-1)
                temp = torch.squeeze(self.ms_model(x_feat)).sigmoid()
                res.append(temp)
            if len(res) > 0:
                pred_sig[pred_sig_obj[0] > thres] = torch.stack(res).to(dtype=torch.float16)
            """"""
            obj_preds = pred_sig_obj[0] > thres
            if len(pred_sig[obj_preds]) > 0:
                prev_dtype = pred_scores.dtype
                pred_scores[obj_preds] = (pred_scores[obj_preds]*8 + self.ms_model(pred_sig[obj_preds].float()).to(dtype=prev_dtype)*2)/10
"""
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        batch['cls_nhot'] = batch['cls_nhot'].to(batch['bboxes'].device)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes'], batch['cls_nhot']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])

        gt_labels, gt_bboxes, gt_labels_nhot = targets.split((1, 4, batch['cls_nhot'].shape[1]), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt, gt_labels_nhot)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        # req loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        loss_sum = loss.sum() * batch_size

        #if self.req_loss:
        #    loss[3] = torch.sum(obj_preds) * batch_size

        return loss_sum, loss.detach()  # loss(box, cls, dfl)

def toGraph(wcnf_plus):
    edge_index = []
    edge_attr = []
    edge_features = [[[1,0], [-1,0]], [[0,1], [0,-1]]]
    mask = np.asarray([1 for _ in range(wcnf_plus.nv)] + [0 for _ in range(len(wcnf_plus.hard))])


    for i, clause in enumerate(wcnf_plus.hard):
        for literal in clause:
            if literal < 0:
                edge_attr.append(edge_features[1][0])
                edge_attr.append(edge_features[1][1])
                edge_index.append([-literal - 1, wcnf_plus.nv + i])
                edge_index.append([wcnf_plus.nv + i, -literal - 1])
            else:
                edge_attr.append(edge_features[0][0])
                edge_attr.append(edge_features[0][1])
                edge_index.append([literal - 1, wcnf_plus.nv + i])
                edge_index.append([wcnf_plus.nv+ i, literal - 1])
    
    edge_index = np.transpose(np.asarray(edge_index))
    edge_attr = np.asarray(edge_attr)
    return edge_attr, edge_index, mask


class CustomTaskAlignedAssigner(TaskAlignedAssigner):

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, gt_labels_nhot):
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt, gt_labels_nhot)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask, gt_labels_nhot)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, gt_labels_nhot):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt, gt_labels_nhot)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps
    
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, gt_labels_nhot):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = (torch.sum(pd_scores[ind[0]] * gt_labels_nhot.unsqueeze(-2).repeat(1,1,na,1), dim=-1).to(dtype=torch.float16))[mask_gt]
        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    
    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask, gt_labels_nhot):

        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w) -> Per anchor, the best target label is decided.
        # 16*14

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        gt_labels_nhot = gt_labels_nhot.to(target_gt_idx.device)
        gt_labels_nhot = gt_labels_nhot.view(-1, gt_labels_nhot.size(-1))
        target_scores = gt_labels_nhot[target_gt_idx]

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores