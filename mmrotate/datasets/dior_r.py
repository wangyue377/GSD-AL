# # Copyright (c) OpenMMLab. All rights reserved.
# import glob
# import os
# import os.path as osp
# import re
# import tempfile
# import time
# import zipfile
# from collections import defaultdict
# from functools import partial
#
# import mmcv
# import numpy as np
# import torch
# from mmcv.ops import nms_rotated
# from mmdet.datasets.custom import CustomDataset
#
# from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
# from .builder import ROTATED_DATASETS
#
#
# @ROTATED_DATASETS.register_module()
# class DIORDataset(CustomDataset):
#     """DIOR-R dataset for rotated object detection with correct class indices.
#
#     Args:
#         ann_file (str): Annotation file path.
#         pipeline (list[dict]): Processing pipeline.
#         version (str, optional): Angle representations ('oc' or 'le90'). Defaults to 'oc'.
#         difficulty (int, optional): Difficulty threshold for GT filtering (0-2). Defaults to 2.
#     """
#     # DIOR-R官方类别列表（索引0-19）
#     CLASSES = (
#         'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
#         'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield',
#         'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
#         'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
#     )
#
#     # 对应类别颜色（20类，与CLASSES顺序一致）
#     PALETTE = [
#         (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
#         (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
#         (128, 0, 128), (0, 128, 128), (255, 128, 0), (128, 255, 0), (0, 128, 255),
#         (255, 0, 128), (128, 255, 128), (255, 128, 128), (128, 255, 255), (255, 255, 255)
#     ]
#
#     def __init__(self,
#                  X_L_file,
#                  label_type,
#                  load_type,
#                  ann_file,
#                  pipeline,
#                  version='oc',
#                  difficulty=2,  # 过滤difficulty>2的样本（DIOR-R困难度0-2）
#                  **kwargs):
#         self.version = version
#         self.difficulty = difficulty
#         self.X_L_file = X_L_file
#         self.label_type = label_type
#         self.load_type = load_type
#
#         super(DIORDataset, self).__init__(X_L_file, label_type, load_type, ann_file, pipeline, **kwargs)
#
#     def load_annotations(self, ann_folder, X_L_file, load_type):
#         """加载DIOR-R数据集标注（8点多边形格式）"""
#         cls_map = {c: i for i, c in enumerate(self.CLASSES)}  # 类别名转索引（0-19）
#         data_infos = []
#
#         # 处理训练/验证/测试集
#         if load_type in ['train', 'val']:
#             img_ids = mmcv.list_from_file(X_L_file)
#             ann_files = [osp.join(ann_folder, f"{img_id}.txt") for img_id in img_ids]
#         else:
#             ann_files = glob.glob(osp.join(ann_folder, "*.txt"))
#
#         for ann_file in ann_files:
#             data_info = {
#                 'filename': '',
#                 'ann': {
#                     'bboxes': [],
#                     'labels': [],
#                     'polygons': []
#                 }
#             }
#             img_id = osp.splitext(osp.basename(ann_file))[0]
#             data_info['filename'] = f"{img_id}.jpg"  # 假设图像为JPEG格式
#
#             with open(ann_file, 'r') as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     parts = line.strip().split()
#                     if len(parts) < 9:
#                         continue
#
#                     # 解析8点多边形坐标和属性
#                     poly = np.array(parts[:8], dtype=np.float32)
#                     cls_name = parts[8]
#                     difficulty = int(parts[9]) if len(parts) > 9 else 0  # DIOR-R困难度0-2
#
#                     # 过滤困难样本和未知类别
#                     if difficulty > self.difficulty or cls_name not in cls_map:
#                         continue
#
#                     # 转换为旋转框（x, y, w, h, angle）
#                     try:
#                         obb = poly2obb_np(poly, self.version)
#                     except Exception as e:
#                         print(f"Invalid polygon in {ann_file}: {poly}, error: {e}")
#                         continue
#
#                     data_info['ann']['polygons'].append(poly)
#                     data_info['ann']['bboxes'].append(obb)
#                     data_info['ann']['labels'].append(cls_map[cls_name])
#
#             # 转换为numpy数组
#             if data_info['ann']['labels']:
#                 data_info['ann']['bboxes'] = np.array(data_info['ann']['bboxes'], dtype=np.float32)
#                 data_info['ann']['labels'] = np.array(data_info['ann']['labels'], dtype=np.int64)
#                 data_info['ann']['polygons'] = np.array(data_info['ann']['polygons'], dtype=np.float32)
#             else:
#                 data_info['ann']['bboxes'] = np.zeros((0, 5), dtype=np.float32)
#                 data_info['ann']['labels'] = np.array([], dtype=np.int64)
#                 data_info['ann']['polygons'] = np.zeros((0, 8), dtype=np.float32)
#
#             data_infos.append(data_info)
#
#         self.img_ids = [d['filename'][:-4] for d in data_infos]  # 存储图像ID（不含.jpg）
#         return data_infos
#
#     def _filter_imgs(self):
#         """过滤无标注样本"""
#         return [i for i, d in enumerate(self.data_infos) if d['ann']['labels'].size > 0]
#
#     def _set_group_flag(self):
#         """设置图像分组标志（默认全为0）"""
#         self.flag = np.zeros(len(self), dtype=np.uint8)
#
#     def evaluate(self, results, metric='mAP', **kwargs):
#         """评估指标（支持mAP计算）"""
#         annotations = [self.get_ann_info(i) for i in range(len(self))]
#         if metric == 'mAP':
#             return eval_rbbox_map(
#                 results,
#                 annotations,
#                 dataset=self.CLASSES,
#                 **kwargs
#             )
#         raise NotImplementedError(f"Unsupported metric: {metric}")
#
#     def merge_det(self, results, nproc=4):
#         """合并分块检测结果（支持DIOR-R分块命名规则）"""
#         collector = defaultdict(list)
#         for idx in range(len(self)):
#             result = results[idx]
#             img_id = self.img_ids[idx]
#             parts = img_id.split('__')  # 假设分块ID为原图名__x__y
#
#             # 解析原图ID和分块坐标
#             if len(parts) > 1 and '__' in img_id:
#                 oriname = parts[0]
#                 x = int(parts[1].split('_')[0])  # 兼容DIOR-R可能的下划线分隔
#                 y = int(parts[2].split('.')[0])
#             else:
#                 oriname = img_id.split('.')[0]
#                 x, y = 0, 0  # 非分块图像坐标为(0,0)
#
#             # 转换分块结果到原图坐标系
#             for cls_idx, cls_dets in enumerate(result):
#                 if cls_dets.size == 0:
#                     continue
#                 bboxes = cls_dets[:, :-1]  # 前5列为旋转框坐标
#                 scores = cls_dets[:, -1]
#                 bboxes[:, :2] += np.array([x, y])  # 调整中心点坐标
#                 collector[oriname].append(
#                     np.hstack([np.full((len(bboxes), 1), cls_idx), bboxes, scores.reshape(-1, 1)])
#                 )
#
#         # 多进程NMS去重
#         merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
#         if nproc <= 1:
#             merged = [merge_func((k, v)) for k, v in collector.items()]
#         else:
#             merged = mmcv.track_parallel_progress(merge_func, list(collector.items()), nproc)
#
#         return zip(*merged)
#
#     def format_results(self, results, submission_dir=None, nproc=4):
#         """生成DIOR-R评估所需的提交文件（含类别索引映射）"""
#         if submission_dir is None:
#             submission_dir = tempfile.TemporaryDirectory().name
#         os.makedirs(submission_dir, exist_ok=True)
#
#         # 合并分块检测结果
#         id_list, dets_list = self.merge_det(results, nproc)
#
#         # 写入各分类结果文件
#         for img_id, dets in zip(id_list, dets_list):
#             for cls_idx, cls_dets in enumerate(dets):
#                 if cls_dets.size == 0:
#                     continue
#                 cls_name = self.CLASSES[cls_idx]
#                 file_path = osp.join(submission_dir, f"Task1_{cls_name}.txt")
#
#                 # 转换为多边形坐标并写入
#                 with open(file_path, 'a') as f:
#                     for det in cls_dets:
#                         obb = det[:5]
#                         score = det[5]
#                         poly = obb2poly_np(obb[np.newaxis, :], self.version)[0]
#                         coords = ' '.join([f"{p:.2f}" for p in poly])
#                         f.write(f"{img_id} {score:.2f} {coords}\n")
#
#         # 压缩提交文件（DIOR-R要求zip格式）
#         zip_path = osp.join(submission_dir, "submission.zip")
#         with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
#             for f in os.listdir(submission_dir):
#                 zf.write(osp.join(submission_dir, f), f)
#
#         return zip_path, submission_dir
#
#
# # 复用DOTA的NMS合并函数
# def _merge_func(info, CLASSES, iou_thr):
#     """合并检测结果并执行旋转框NMS"""
#     img_id, dets = info
#     if not dets:
#         return img_id, [np.zeros((0, 6)) for _ in CLASSES]
#
#     dets = np.vstack(dets)
#     labels = dets[:, 0].astype(int)
#     bboxes = dets[:, 1:-1]  # (x, y, w, h, angle)
#     scores = dets[:, -1]
#
#     results = []
#     for cls_idx in range(len(CLASSES)):
#         mask = (labels == cls_idx)
#         if not np.any(mask):
#             results.append(np.zeros((0, 6), dtype=np.float32))
#             continue
#
#         # 转换为Tensor进行NMS
#         cls_bboxes = torch.from_numpy(bboxes[mask]).float()
#         cls_scores = torch.from_numpy(scores[mask]).float()
#         keep = nms_rotated(cls_bboxes, cls_scores, iou_thr=iou_thr)
#
#         # 合并结果（标签+旋转框+分数）
#         cls_results = torch.cat([
#             torch.full((len(keep), 1), cls_idx, dtype=torch.float32),
#             cls_bboxes[keep],
#             cls_scores[keep].unsqueeze(1)
#         ], dim=1).numpy()
#         results.append(cls_results)
#
#     return img_id, results


# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import zipfile
from collections import defaultdict
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class DIORDataset(CustomDataset):
    """DOTA dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """

    # DIOR-R官方类别列表（索引0-19）
    CLASSES = (
        'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
        'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield',
        'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
        'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
    )

    # 对应类别颜色（20类，与CLASSES顺序一致）
    PALETTE = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128), (255, 128, 0), (128, 255, 0), (0, 128, 255),
        (255, 0, 128), (128, 255, 128), (255, 128, 128), (128, 255, 255), (255, 255, 255)
    ]

    def __init__(self,
                 X_L_file,
                 label_type,
                 load_type,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty
        self.X_L_file = X_L_file
        self.label_type = label_type
        self.load_type = load_type

        super(DIORDataset, self).__init__(X_L_file, label_type, load_type, ann_file, pipeline, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_folder, X_L_file, load_type):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based

        ann_files_ori = glob.glob(ann_folder + '*.txt')

        if not ann_files_ori:
            ann_files = []
        else:
            # train phase:normal
            img_ids = mmcv.list_from_file(self.X_L_file)
            ann_files = []
            for i in img_ids:
                ann_files.append(ann_folder + i + ".txt")

        data_infos = []
        if not ann_files:  # test phase
            if load_type == 'select':
                img_ids = mmcv.list_from_file(self.X_L_file)
                ann_files = []
                for i in img_ids:
                    ann_files.append(ann_folder + i + ".txt")
            else:
                ann_files = glob.glob(ann_folder + '/*.jpg')
                # ann_files = sorted(glob.glob(ann_folder + '/*.png'))
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        cls_name = bbox_info[8]
                        if cls_name == "background":
                            continue
                        difficulty = int(bbox_info[9])
                        label = cls_map[cls_name]
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if data_info['ann']['labels'].size > 0:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.
        """
        collector = defaultdict(list)
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        return zip(*merged_results)

    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Args:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        if osp.exists(out_folder):
            raise ValueError(f'The out_folder should be a non-exist path, '
                             f'but {out_folder} is existing')
        os.makedirs(out_folder)

        files = [
            osp.join(out_folder, 'Task1_' + cls + '.txt')
            for cls in self.CLASSES
        ]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0:
                    continue
                bboxes = obb2poly_np(dets, self.version)
                for bbox in bboxes:
                    txt_element = [img_id, str(bbox[-1])
                                   ] + [f'{p:.2f}' for p in bbox[:-1]]
                    f.writelines(' '.join(txt_element) + '\n')

        for f in file_objs:
            f.close()

        target_name = osp.split(out_folder)[-1]
        with zipfile.ZipFile(
                osp.join(out_folder, target_name + '.zip'), 'w',
                zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])

        return files

    def format_results(self, results, submission_dir=None, nproc=4, **kwargs):
        """Format the results to submission text (standard format for DOTA
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            submission_dir (str, optional): The folder that contains submission
                files. If not specified, a temp folder will be created.
                Default: None.
            nproc (int, optional): number of process.

        Returns:
            tuple:

                - result_files (dict): a dict containing the json filepaths
                - tmp_dir (str): the temporal directory created for saving \
                    json files when submission_dir is not specified.
        """
        nproc = min(nproc, os.cpu_count())
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to '
            f'the dataset len: {len(results)} != {len(self)}')
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None

        print('\nMerging patch bboxes into full image!!!')
        start_time = time.time()
        id_list, dets_list = self.merge_det(results, nproc)
        stop_time = time.time()
        print(f'Used time: {(stop_time - start_time):.1f} s')

        result_files = self._results2submission(id_list, dets_list,
                                                submission_dir)

        return result_files, tmp_dir


def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Args:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                              iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results
