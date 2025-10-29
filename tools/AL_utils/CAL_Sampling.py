import torch
import pickle
import time
from tqdm import tqdm
import copy
import numpy as np
from mmrotate.core.bbox.iou_calculators.rotate_iou2d_calculator import RBboxOverlaps2D
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import os


def calculate_iou(box1, box2):
    """
    使用新的IoU计算逻辑计算两个定向边界框的IoU
    :param box1: 预测框，形状为 (5,) 或 (6,)，格式为 <cx, cy, w, h, a> 或 <cx, cy, w, h, a, score>
    :param box2: 真实框，形状为 (5,) 或 (6,)，格式为 <cx, cy, w, h, a> 或 <cx, cy, w, h, a, score>
    :return: IoU值
    """
    if box2 is None:
        return 0
    iou_calculator = RBboxOverlaps2D()
    box1 = box1.unsqueeze(0)
    box2 = box2.unsqueeze(0)
    iou = iou_calculator(box1, box2, mode='iou', is_aligned=True).item()
    return iou


# 假设 load_data_to_gpu 已经正确定义
def load_data_to_gpu(data):
    for key in data:
        if torch.is_tensor(data[key]):
            data[key] = data[key].cuda()


class CAL_Sampling_Scene:
    def __init__(self, cfg):
        self.cfg = cfg
        self.class_weight = getattr(cfg.ALSSL_TRAIN, 'SAMPLE_CLASSWEIGHT', [1, 1, 1])
        self.s_weight = getattr(cfg.ALSSL_TRAIN, 'S_CLASSWEIGHT', [1, 1, 1])
        self.uncer_class = getattr(cfg.ALSSL_TRAIN, 'UNCER_CLASS', False)
        self.sample_weight = getattr(cfg.ALSSL_TRAIN, 'SAMPLE_WEIGHT', [5, 1])
        assert sum(self.class_weight) == len(self.class_weight), 'sum error ! should equal to class numbers'
        self.class_ensemble_thresh = getattr(cfg.ALSSL_TRAIN, 'CLASS_ENSEMBLE_THRESH', [0.8, 0.8, 0.8])
        self.class_ensemble_iou_thresh = getattr(cfg.ALSSL_TRAIN, 'CLASS_ENSEMBLE_iou_THRESH', 0.5)
        self.reliable_iou_thresh = getattr(cfg.ALSSL_TRAIN, 'Reliable_iou_THRESH', 0.2)

    def calc_similarity(self, feat1, feat_list):
        if len(feat_list) == 0 or feat1.sum() == 0:
            return 0
        div_mx = 0
        for feat in feat_list:
            similarity_matrix = pairwise_cosine_similarity(feat1, feat)
            sim_mx = similarity_matrix.max(1)[0].mean()
            div_mx = max(div_mx, sim_mx)
        return div_mx

    def find_reliable_boxes(self, pred_boxes, pred_boxes_list):
        for pred_boxes_mmodes in pred_boxes_list:
            if pred_boxes.shape[0] == 0:
                return []
            if pred_boxes_mmodes.shape[0] == 0:
                pred_boxes_mmodes = pred_boxes
            iou = calculate_iou(pred_boxes, pred_boxes_mmodes)
            box_match = iou.max(axis=1) > self.reliable_iou_thresh
        return box_match

    def ensemble_boxes(self, pred_boxes, pred_scores, pred_labels, pred_boxes_list, pred_scores_list, pred_labels_list):
        thresh = self.class_ensemble_thresh
        iou_thresh = self.class_ensemble_iou_thresh
        o_pred_boxes = pred_boxes_list[0]
        o_pred_scores = pred_scores_list[0]
        o_pred_labels = pred_labels_list[0]

        box_flag = o_pred_scores > 0.5
        o_pred_boxes = o_pred_boxes[box_flag]
        o_pred_scores = o_pred_scores[box_flag]
        o_pred_labels = o_pred_labels[box_flag]
        iou = calculate_iou(o_pred_boxes, pred_boxes)
        if o_pred_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
            return pred_boxes, pred_scores, pred_labels
        for o_idx in range(o_pred_boxes.shape[0]):
            o_pred_box = o_pred_boxes[o_idx]
            o_pred_score = o_pred_scores[o_idx]
            o_pred_label = o_pred_labels[o_idx]
            o_iou = iou[o_idx]
            if o_iou.max() > iou_thresh:
                p_idx = o_iou.argmax()
                if o_pred_score > pred_scores[p_idx]:
                    pred_boxes[p_idx] = o_pred_box
                    pred_scores[p_idx] = o_pred_score
                    pred_labels[p_idx] = o_pred_label
            elif o_pred_score > thresh[o_pred_label - 1]:
                pred_boxes = torch.cat((pred_boxes, o_pred_box.unsqueeze(0)))
                pred_scores = torch.cat((pred_scores, o_pred_score.unsqueeze(0)))
                pred_labels = torch.cat((pred_labels, o_pred_label.unsqueeze(0)))
        return pred_boxes, pred_scores, pred_labels


def ssl_select(X_U, budget, iou_thresh, data_loader, model, cfg, label_num=0, class_num=15, score_thresh=None):
    """
    实现主动学习采样策略
    :param X_U: 未标注数据的索引列表
    :param budget: 主动学习预算，即每次选择的样本数量
    :param iou_thresh: IoU阈值，用于筛选有效预测框
    :param data_loader: 数据加载器
    :param model: 目标检测模型
    :param cfg: 配置参数
    :param label_num: 标签数量
    :param class_num: 类别数量
    :param score_thresh: 分数阈值，用于筛选有效预测框
    :return: 选择的样本索引列表
    """
    model.eval()
    model.cuda(cfg.gpu_ids[0])
    sampler = CAL_Sampling_Scene(cfg)

    select_dic = {}
    unlabel_data_boxes = {}
    pred_res = []
    pred_path = getattr(cfg.ALSSL_TRAIN, 'ENSEMBLE_RESULT', None)
    if pred_path is not None:
        if type(pred_path) == list:
            for pred_p in pred_path:
                with open(pred_p, 'rb') as f:
                    pred_res.append(pickle.load(f))

    total_it_each_epoch = len(data_loader)
    pbar = tqdm(total=total_it_each_epoch, leave=True, desc='evaluating_unlabelled_set', dynamic_ncols=True)

    print('-----------------------')
    print('begin')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    pred_record = {}
    for cur_it in range(total_it_each_epoch):
        try:
            unlabelled_batch = next(iter(data_loader))
        except StopIteration:
            unlabelled_dataloader_iter = iter(data_loader)
            unlabelled_batch = next(unlabelled_dataloader_iter)
        with torch.no_grad():
            load_data_to_gpu(unlabelled_batch)
            pred_dicts, _ = model(unlabelled_batch)
            pred_dicts_list = []
            for pred_r in pred_res:
                if unlabelled_batch['frame_id'][0] in pred_r:
                    pred_dicts_list.append(pred_r[unlabelled_batch['frame_id'][0]])

            for batch_inx in range(len(pred_dicts)):
                unlabel_data_boxes[unlabelled_batch['frame_id'][batch_inx]] = [
                    len(unlabelled_batch['gt_names'][0][batch_inx]),
                    unlabelled_batch['gt_names'][0][batch_inx],
                    unlabelled_batch['gt_boxes']
                ]

                pred_boxes = pred_dicts[batch_inx]['pred_boxes']
                pred_scores = pred_dicts[batch_inx]['pred_scores']
                pred_labels = pred_dicts[batch_inx]['pred_labels']
                pred_record[unlabelled_batch['frame_id'][batch_inx]] = {
                    'pred_boxes': pred_boxes,
                    'pred_scores': pred_scores,
                    'pred_labels': pred_labels
                }

                pred_boxes_list = []
                pred_scores_list = []
                pred_labels_list = []
                for pred_dict_models in pred_dicts_list:
                    pred_boxes_list.append(pred_dict_models['pred_boxes'])
                    pred_scores_list.append(pred_dict_models['pred_scores'])
                    pred_labels_list.append(pred_dict_models['pred_labels'])

                no_box = False
                pred_boxes_r = pred_boxes
                pred_scores_r = pred_scores
                pred_labels_r = pred_labels
                if len(pred_boxes_list) > 0:
                    cls_ens = cfg.ALSSL_TRAIN.get('CLS_ENSEM', True)
                    pred_boxes, pred_scores, pred_labels = sampler.ensemble_boxes(pred_boxes, pred_scores, pred_labels,
                                                                                  pred_boxes_list, pred_scores_list,
                                                                                  pred_labels_list)
                    reliable_boxes_idx = sampler.find_reliable_boxes(pred_boxes, pred_boxes_list)
                    if len(reliable_boxes_idx) > 0 and cls_ens:
                        pred_boxes_r = pred_boxes[reliable_boxes_idx]
                        pred_scores_r = pred_scores[reliable_boxes_idx]
                        pred_labels_r = pred_labels[reliable_boxes_idx]
                    else:
                        no_box = True

                if score_thresh is not None:
                    selected_box = (pred_scores > score_thresh[1]) & (pred_scores < score_thresh[0])
                    pred_boxes = pred_boxes[selected_box]
                    pred_scores = pred_scores[selected_box]
                    pred_labels = pred_labels[selected_box]

                box_values = ((-pred_scores * torch.log2(pred_scores)) - (
                            (1 - pred_scores) * torch.log2(1 - pred_scores)))
                box_values = box_values * box_values.new_tensor([sampler.s_weight[cl.item() - 1] for cl in pred_labels])
                if sampler.uncer_class:
                    box_values = box_values * box_values.new_tensor(
                        [sampler.class_weight[cl.item() - 1] for cl in pred_labels])

                if cfg.ALSSL_TRAIN.AGGREGATION == 'mean':
                    aggregated_values = torch.mean(box_values)
                    if box_values.numel() == 0 or no_box:
                        aggregated_values = box_values.new_zeros(1)
                else:
                    raise NotImplementedError

                shared_features = pred_dicts[0]['shared_features']
                pred_boxes_emb = pred_dicts[0]['box_features']
                select_dic[unlabelled_batch['frame_id'][batch_inx]] = [
                    aggregated_values,
                    shared_features,
                    pred_labels_r,
                    pred_scores_r,
                    pred_dicts[0]['box_features'],
                    box_values,
                    0,
                    pred_boxes,
                    pred_labels,
                    pred_scores,
                    pred_boxes_r
                ]
        pbar.update()
        pbar.refresh()
    pbar.close()

    # sort and get selected_frames
    select_dic = dict(sorted(select_dic.items(), key=lambda item: item[1][0]))
    unlabelled_sample_num = len(select_dic.keys())
    selected_frames = list(select_dic.keys())[::-1]
    selected_items = list(select_dic.values())[::-1]

    embedding_list = [select_dic[f][1] for f in selected_frames]
    embeddings = torch.stack(embedding_list, 0)
    num_sample = embeddings.shape[0]
    embeddings = embeddings.view(num_sample, -1)

    thresh = cfg.ALSSL_TRAIN.get('SELECT_BOX_NUMS', 0)
    div_thresh = cfg.ALSSL_TRAIN.get('DIV_THRESH', 0.9)
    obj_thresh = cfg.ALSSL_TRAIN.get('OBJ_THRESH', [0.1, 0.1, 0.1])
    class_names = getattr(cfg.DATA_CONFIG, 'CLASS_NAMES', [])
    if len(class_names) == 1:
        cl_thresh = [thresh]
        cl_sample_num = [0]
        cl_flag = [True]
        gt_sample_num = [0]
    elif len(class_names) == 3:
        cl_thresh = [sampler.class_weight[c_id] * (thresh / 3) + 1 for c_id in range(len(sampler.class_weight))]
        cl_sample_num = [0, 0, 0]
        cl_flag = [True, True, True]
        gt_sample_num = [0, 0, 0]

    selected_frames_by_boxes = []
    selected_feats = []

    if thresh > 0:
        s = 0
        cnt = 1
        d_cnt = 0
        while s < thresh:
            cnt = 1
            for idx, frame_id in enumerate(selected_frames):
                if s >= thresh:
                    break
                num_boxes = unlabel_data_boxes[frame_id][0]
                gt_names = unlabel_data_boxes[frame_id][1]
                gt_boxes = unlabel_data_boxes[frame_id][2]

                if len(gt_names) == 0:
                    continue
                item = selected_items[idx]
                box_feats = item[4]
                div = sampler.calc_similarity(box_feats, selected_feats)
                if div > div_thresh * cnt:
                    continue
                pred_boxes = item[7]
                box_labels = item[8]
                box_scores = item[9]
                box_labels_r = item[2]
                box_scores_r = item[3]
                pred_boxes_r = item[10]

                flag = False
                flag_2 = False
                flag_3 = False
                for cl in range(len(cl_sample_num)):
                    if cl_sample_num[cl] >= cl_thresh[cl]:
                        cl_flag[cl] = False
                        continue
                    labels_flag = (cl + 1) == box_labels_r
                    cl_num = (box_scores_r[labels_flag] > obj_thresh[cl] / cnt).sum()
                    if cl_num + cl_sample_num[cl] >= cl_thresh[cl]:
                        cl_flag[cl] = False
                        flag_2 = True
                    if cl_num > 0:
                        flag = True
                        break
                if not flag:
                    continue
                if 'Kitti' in cfg.DATA_CONFIG.DATASET.strip('Dataset'):
                    labeles = [1 for obj in data_loader.dataset.get_label(frame_id) if obj.cls_type == 'DontCare']
                    dc_num = len(labeles)
                    if dc_num >= 1 * cnt:
                        d_cnt += 1
                        continue

                s += num_boxes
                for cl in range(len(cl_sample_num)):
                    labels_flag = (cl + 1) == box_labels
                    cl_num = (box_scores[labels_flag] > obj_thresh[cl] / cnt).sum()
                    cl_sample_num[cl] = cl_sample_num[cl] + cl_num
                for cl_idx, cl_name in enumerate(class_names):
                    cl_num_ = (cl_name == gt_names).sum()
                    gt_sample_num[cl_idx] += cl_num_
                selected_frames_by_boxes.append(frame_id)
                selected_feats.append(box_feats)
                del select_dic[frame_id]
                if flag_2:
                    break

            for frame_id, item in select_dic.items():
                pred_labels = item[8]
                box_values = copy.deepcopy(item[5])

                for cl_idx in range(len(class_names)):
                    box_cl_flag = pred_labels == (cl_idx + 1)
                    if cl_flag[cl_idx] == False:
                        box_values[box_cl_flag] = box_values[box_cl_flag] * (1. / sampler.sample_weight[0])
                if box_values.numel() == 0:
                    aggregated_values = box_values.new_zeros(1)
                elif cfg.ALSSL_TRAIN.AGGREGATION == 'mean':
                    aggregated_values = torch.mean(box_values)
                    if box_values.numel() == 0 or no_box:
                        aggregated_values = box_values.new_zeros(1)
                else:
                    raise NotImplementedError
                item[0] = aggregated_values
                item[5] = box_values
                select_dic[frame_id] = item

            select_dic = dict(sorted(select_dic.items(), key=lambda item: item[1][0]))

            selected_frames = list(select_dic.keys())[::-1]
            selected_items = list(select_dic.values())[::-1]
            cnt += 0.1

            print('DC Count %d' % d_cnt)
            print('----------cl_sample_num--------')
            print(cl_flag)
            print(cl_sample_num)
        selected_frames = selected_frames_by_boxes
