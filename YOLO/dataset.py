from ultralytics.data.dataset import YOLODataset, DATASET_CACHE_VERSION
from ultralytics.data.utils import get_hash, img2label_paths, HELP_URL, exif_size
from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, LOGGER, is_dir_writeable
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import segments2boxes

from copy import deepcopy
from pathlib import Path
import numpy as np
from multiprocessing.pool import ThreadPool
from itertools import repeat
import torch
from PIL import Image, ImageOps
import os

class ROADYOLODataset(YOLODataset):

    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, lb_class_id={}, lb_id_class={}, lb_id_class_norm={}, **kwargs):
        self.lb_class_id = lb_class_id
        self.lb_id_class = lb_id_class
        self.lb_id_class_norm = lb_id_class_norm
        super().__init__(*args, data=data, use_segments=use_segments, use_keypoints=use_keypoints, **kwargs)

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            (cache, cache2, cache3, cache4), exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            self.lb_class_id = cache2
            self.lb_id_class = cache3
            self.lb_id_class_norm = cache4
            assert cache['version'] == DATASET_CACHE_VERSION  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops
        
        for i in range(41):
            self.lb_id_class[i] = np.identity(41)[i]

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        if not labels:
            LOGGER.warning(f'WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}')
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            LOGGER.warning(f'WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}')
        return labels

    def cache_labels(self, path=Path('./labels.cache')):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.im_files)
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        lb_class_id = {}
        lb_id_class = {}
        lb_id_class_norm = {}

        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim)))
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg, lb_class in pbar:
                if msg != '':
                    print(msg)
                    exit()
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    lb_class_new = []
                    for lb_class_single in lb_class:
                        if len(lb_class_single) > 1:
                            lb_class_single_str = ','.join(list(map(str,lb_class_single)))

                            if not lb_class_single_str in lb_class_id:
                                lb_class_id[lb_class_single_str] = len(lb_class_id) + 41
                                lb_class_single_np = np.sum(np.identity(41)[lb_class_single], axis=0)
                                lb_id_class[len(lb_id_class) + 41] = lb_class_single_np
                                lb_id_class_norm[len(lb_id_class_norm) + 41] = lb_class_single
                                
                            lb_class_new.append(lb_class_id[lb_class_single_str])
                        else:
                            lb_class_new.append(lb_class_single[0])
                    lb_class_new = np.array(lb_class_new, dtype=np.float32)
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb_class_new.reshape(-1,1),  # n, 1
                            bboxes=lb,  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}.')
        print("Finished!")
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        self.lb_class_id.update(lb_class_id)
        self.lb_id_class.update(lb_id_class)
        self.lb_id_class_norm.update(lb_id_class_norm)

        save_dataset_cache_file(self.prefix, path, x, self.lb_class_id, self.lb_id_class, self.lb_id_class_norm)
        return x

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        l1 = self.get_image_and_label(index)
        l2 = self.transforms(l1)
        cls_nhot = []
        for cls_single in l2['cls']:
            cls_nhot.append(self.lb_id_class[int(cls_single)])
        if cls_nhot == []:
            cls_nhot = np.zeros((0,41), dtype=np.float32)
        else:
            cls_nhot = np.array(cls_nhot, dtype=np.float32)
        l2['cls_nhot'] = torch.from_numpy(cls_nhot)
        return l2

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        if self.rect:
            label['rect_shape'] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)
    
    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):

            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls', 'cls_nhot']:
                value = torch.cat(value, 0)
            new_batch[k] = value

        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)

        return new_batch
    
    def update_labels_info(self, label):
        """custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc
    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path) + "_v2.npy", allow_pickle=True).item()  # load dict
    cache2 = np.load(str(path) + "_labels.npy", allow_pickle=True).item()  # load dict
    cache3 = np.load(str(path) + "_labels2.npy", allow_pickle=True).item()  # load dict
    cache4 = np.load(str(path) + "_labels3.npy", allow_pickle=True).item()  # load dict
    gc.enable()
    return cache, cache2, cache3, cache4

def save_dataset_cache_file(prefix, path, x, lb_class_id, lb_id_class, lb_id_class_norm):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x['version'] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path) + "_v2", x)  # save cache for next time
        np.save(str(path) + "_labels", lb_class_id)
        np.save(str(path) + "_labels2", lb_id_class)
        np.save(str(path) + "_labels3", lb_id_class_norm)
        LOGGER.info(f'{prefix}New cache created: {path}')
    else:
        LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')


def verify_image_label(args):
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, '', [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb_classes = []
                lb_max = []
                lb_class_first = []
                for i in range(len(lb)):
                    lb_now = list(map(int,lb[i].pop(0).split(",")))
                    lb_classes.append(lb_now)
                    lb_class_first.append([lb_now[0]])
                    lb_max.append(max(lb_classes[-1]))
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
                lb_max = np.array(lb_max, dtype=np.float32)
                lb_class_first = np.array(lb_class_first, dtype=np.float32)

                
            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f'labels require {(5 + nkpt * ndim)} columns each'
                    assert (lb[:, 5::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                    assert (lb[:, 6::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                else:
                    assert lb.shape[1] == 4, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb <= 1).all(), \
                        f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                # All labels
                max_cls = int(lb_max.max())  # max label count
                assert max_cls <= num_cls, \
                    f'Label class {max_cls} exceeds dataset class count {num_cls}. ' \
                    f'Possible class labels are 0-{num_cls - 1}'
                _, i = np.unique(np.concatenate((lb_class_first, lb), axis=1), axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    lb_classes = np.array(lb_classes)[i].tolist()
                    if segments:
                        segments = [segments[x] for x in i]
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (4 + nkpt * ndim)), dtype=np.float32) if keypoint else np.zeros(
                    (0, 4), dtype=np.float32)
                lb_classes = np.zeros((0, 1), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (4 + nkpt * ndim)), dtype=np.float32) if keypoint else np.zeros((0, 4), dtype=np.float32)
            lb_classes = np.zeros((0, 1), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg, lb_classes
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, None, nm, nf, ne, nc, msg, None]