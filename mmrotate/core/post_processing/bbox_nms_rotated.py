# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import nms_rotated
from mmcv.ops import box_iou_rotated


# def multiclass_nms_rotated(multi_bboxes,
#                            multi_scores,
#                            score_thr,
#                            nms,
#                            stage1_rois,
#                            max_num=-1,
#                            score_factors=None,
#                            return_inds=False):
#
#
#     num_classes = multi_scores.size(1) - 1
#     # exclude background category
#     if multi_bboxes.shape[1] > 5:
#         bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
#     else:
#         bboxes = multi_bboxes[:, None].expand(
#             multi_scores.size(0), num_classes, 5)
#
#     # 检查 bboxes 和 scores 的初始设备
#     #print(f"Device of bboxes at the beginning: {bboxes.device}")
#     #print(f"Device of scores at the beginning: {scores.device}")
#     scores = multi_scores[:, :-1]
#
#     # ===================== added:compute bg score ====================
#     if multi_bboxes.shape[1] > 5:
#         bg_scores = multi_scores[:, -1]
#         bg_scores_15 = [bg_scores[i].repeat(int(multi_bboxes.shape[1]/5)) for i in range(bg_scores.size(0))]
#         bg_scores_15_new = torch.Tensor(
#             [item.cpu().detach().numpy() for item in bg_scores_15])
#         bg_scores = bg_scores_15_new.view(-1, 1)
#     else:
#         bg_scores = multi_scores[:, -1]
#     # ===================== added:compute entropy =====================\
#     if multi_bboxes.shape[1] > 5:
#         probs = scores
#         log_probs = torch.log(probs)
#         entropys = (probs * log_probs).sum(1)
#         entropys_15 = [entropys[i].repeat(int(multi_bboxes.shape[1]/5)) for i in range(entropys.size(0))]
#         entropys_15_new = torch.Tensor(
#             [item.cpu().detach().numpy() for item in entropys_15])
#         entropys = entropys_15_new.view(-1, 1)
#     else:
#         probs = scores
#         log_probs = torch.log(probs)
#         entropys = (probs * log_probs).sum(1)
#
#     labels = torch.arange(num_classes, dtype=torch.long)
#     labels = labels.view(1, -1).expand_as(scores)
#     bboxes = bboxes.reshape(-1, 5)
#     scores = scores.reshape(-1)
#     labels = labels.reshape(-1)
#
#     # remove low scoring boxes
#     valid_mask = scores > score_thr
#     if score_factors is not None:
#         # expand the shape to match original shape of score
#         score_factors = score_factors.view(-1, 1).expand(
#             multi_scores.size(0), num_classes)
#         score_factors = score_factors.reshape(-1)
#         scores = scores * score_factors
#
#     inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
#
#     # 打印 inds 的设备信息
#     #print(f"Device of inds before moving: {inds.device}")
#
#     # 假设期望所有张量都在 GPU 上，将 inds 移动到 GPU
#     device = bboxes.device  # 获取 bboxes 的设备
#     inds = inds.to(device)
#     #print(f"Device of inds after moving: {inds.device}")
#
#     # 强制将 bboxes, scores, labels, entropys, bg_scores 移动到同一设备
#     bboxes = bboxes.to(device)
#     scores = scores.to(device)
#     labels = labels.to(device)
#     entropys = entropys.to(device)
#     bg_scores = bg_scores.to(device)
#
#     bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
#     entropys = entropys[inds]
#     bg_scores = bg_scores[inds]
#
#     if bboxes.numel() == 0:
#         dets = torch.cat([bboxes, scores[:, None]], -1)
#         if return_inds:
#             return dets, labels, inds
#         else:
#             return dets, labels, entropys, bg_scores
#
#     max_coordinate = bboxes.max()
#     offsets = labels.to(bboxes) * (max_coordinate + 1)
#     if bboxes.size(-1) == 5:
#         bboxes_for_nms = bboxes.clone()
#         bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
#     else:
#         bboxes_for_nms = bboxes + offsets[:, None]
#     _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)
#
#     if max_num > 0:
#         keep = keep[:max_num]
#
#     keep = keep.to(device)
#     bboxes = bboxes[keep]
#     scores = scores[keep]
#     labels = labels[keep]
#     entropys = entropys[keep]
#     bg_scores = bg_scores[keep]
#
#     if return_inds:
#         return torch.cat([bboxes, scores[:, None]], 1), labels, keep
#     else:
#         return torch.cat([bboxes, scores[:, None]], 1), labels, entropys, bg_scores


def multiclass_nms_rotated(multi_bboxes,
                           multi_scores,
                           score_thr,
                           nms,
                           stage1_rois,
                           max_num=-1,
                           score_factors=None,
                           return_inds=False):
    """
    处理旋转框的多类别NMS，并返回所有类别分数

    Args:
        multi_bboxes (Tensor): 旋转框坐标，形状为 (num_boxes, 5) 或 (num_boxes, num_classes*5)
        multi_scores (Tensor): 全类别分数，形状为 (num_boxes, num_classes+1)（最后一维为背景）
        score_thr (float): 置信度阈值
        nms (dict): NMS配置（包含iou_thr等）
        stage1_rois (Tensor): 阶段1的ROIs（可能用于坐标转换，此处未使用）
        max_num (int): 最多保留的检测框数
        score_factors (Tensor): 分数缩放因子
        return_inds (bool): 是否返回索引

    Returns:
        dets (Tensor): 检测框 (cx, cy, w, h, a, score)
        labels (Tensor): 类别标签（排除背景）
        entropys (Tensor): 类别分布熵值
        bg_scores (Tensor): 背景类分数
        all_scores (Tensor): 每个检测框的全类别分数（shape: [num_det, num_classes+1]）
    """
    num_classes = multi_scores.size(1) - 1  # 总类别数（排除背景）

    # ---------------------- 1. 数据格式转换 ----------------------
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(-1, num_classes, 5)  # (N, C, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 5)  # (N, 1, 5) -> (N, C, 5)

    # ---------------------- 2. 计算背景分数和熵值 ----------------------
    bg_scores = multi_scores[:, -1]  # 背景类分数，形状 (N,)
    probs = multi_scores[:, :-1]  # 非背景类分数，形状 (N, C)
    entropys = -(probs * torch.log(probs + 1e-10)).sum(dim=1)  # 熵值，形状 (N,)

    # ---------------------- 3. 展平数据以便处理 ----------------------
    bboxes = bboxes.reshape(-1, 5)  # 展平为 (N*C, 5)
    scores = probs.reshape(-1)  # 非背景类分数展平为 (N*C,)
    labels = torch.arange(num_classes, dtype=torch.long).view(1, -1).expand_as(probs).reshape(-1)  # 类别标签展平为 (N*C,)

    # ---------------------- 4. 过滤低置信度框 ----------------------
    valid_mask = scores > score_thr
    if score_factors is not None:
        scores = scores * score_factors
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)

    if inds.numel() == 0:
        # 处理空检测情况（所有返回值保持一致的空形状）
        empty = torch.empty(0, dtype=torch.float32, device=bboxes.device)
        empty_labels = torch.empty(0, dtype=torch.long, device=bboxes.device)
        empty_scores = torch.empty(0, num_classes + 1, device=bboxes.device)
        return (
            torch.cat([empty, empty[:, None]], -1),  # dets
            empty_labels,  # labels
            empty,  # entropys
            empty,  # bg_scores
            empty_scores,  # all_scores
        )

    # ---------------------- 5. 提取所有需要的张量（保持索引一致） ----------------------
    box_indices = torch.div(inds, num_classes, rounding_mode='floor')  # 原始框索引

    # 提取并统一设备
    bboxes = bboxes[inds]
    # print("bboxes shape:", bboxes.shape)
    scores = scores[inds]
    # print("scores shape:", scores.shape)
    labels = labels.to(inds.device)
    labels = labels[inds]
    # print("labels shape:", labels.shape)
    entropys = entropys[box_indices]
    # print("entropys shape:", entropys.shape)
    bg_scores = bg_scores[box_indices]
    # print("bg_scores shape:", bg_scores.shape)
    all_scores = multi_scores[box_indices]  # 关键：通过box_indices直接提取原始框的全类别分数
    # print("all_scores shape:", all_scores.shape)

    # ---------------------- 6. NMS处理 ----------------------
    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes.clone()
    bboxes_for_nms[:, :2] += offsets[:, None]

    keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)[1]
    if max_num > 0:
        keep = keep[:max_num]

    # ---------------------- 7. 应用NMS结果（保持索引一致） ----------------------
    dets = torch.cat([bboxes[keep], scores[keep, None]], -1)
    labels = labels[keep]
    entropys = entropys[keep]
    bg_scores = bg_scores[keep]
    all_scores = all_scores[keep]  # 关键：与其他张量保持相同的索引操作

    if return_inds:
        return dets, labels, entropys, bg_scores, all_scores, keep
    else:
        return dets, labels, entropys, bg_scores, all_scores


def aug_multiclass_nms_rotated(merged_bboxes, merged_labels, score_thr, nms,
                               max_num, classes):
    """NMS for aug multi-class bboxes.

    Args:
        multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (torch.Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): Config of NMS.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        classes (int): number of classes.

    Returns:
        tuple (dets, labels): tensors of shape (k, 5), and (k). Dets are boxes
            with scores. Labels are 0-based.
    """
    bboxes, labels = [], []

    for cls in range(classes):
        cls_bboxes = merged_bboxes[merged_labels == cls]
        inds = cls_bboxes[:, -1] > score_thr
        if len(inds) == 0:
            continue
        cur_bboxes = cls_bboxes[inds, :]
        cls_dets, _ = nms_rotated(cur_bboxes[:, :5], cur_bboxes[:, -1],
                                  nms.iou_thr)
        cls_labels = merged_bboxes.new_full((cls_dets.shape[0],),
                                            cls,
                                            dtype=torch.long)
        if cls_dets.size()[0] == 0:
            continue
        bboxes.append(cls_dets)
        labels.append(cls_labels)

    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, _inds = bboxes[:, -1].sort(descending=True)
            _inds = _inds[:max_num]
            bboxes = bboxes[_inds]
            labels = labels[_inds]
    else:
        bboxes = merged_bboxes.new_zeros((0, merged_bboxes.size(-1)))
        labels = merged_bboxes.new_zeros((0,), dtype=torch.long)

    return bboxes, labels

# # Copyright (c) OpenMMLab. All rights reserved.
# import torch
# from mmcv.ops import nms_rotated
# from mmcv.ops import box_iou_rotated
#
#
# def multiclass_nms_rotated(multi_bboxes,
#                            multi_scores,
#                            score_thr,
#                            nms,
#                            stage1_rois,
#                            max_num=-1,
#                            score_factors=None,
#                            return_inds=False):
#     """NMS for multi-class bboxes.
#
#     Args:
#         multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)                DOTA:(n, #class*5)  HRSC:(n, 5)   n=2000
#         multi_scores (torch.Tensor): shape (n, #class), where the last column     n:预测框数量(2000)
#             contains scores of the background class, but this will be ignored.
#         score_thr (float): bbox threshold, bboxes with scores lower than it
#             will not be considered.
#         nms (float): Config of NMS.
#         max_num (int, optional): if there are more than max_num bboxes after
#             NMS, only top max_num will be kept. Default to -1.
#         score_factors (Tensor, optional): The factors multiplied to scores
#             before applying NMS. Default to None.
#         return_inds (bool, optional): Whether return the indices of kept
#             bboxes. Default to False.
#
#     Returns:
#         tuple (dets, labels, indices (optional)): tensors of shape (k, 5), \
#         (k), and (k). Dets are boxes with scores. Labels are 0-based.
#     """
#     num_classes = multi_scores.size(1) - 1   # DOTA:15  (0,1,2...,15)  15:backkground   HRSC:1-->(0,1)  0:bckground  1:ship
#     # exclude background category
#     if multi_bboxes.shape[1] > 5:
#         bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
#     else:
#         bboxes = multi_bboxes[:, None].expand(
#             multi_scores.size(0), num_classes, 5)
#     scores = multi_scores[:, :-1]
#
#     # ===================== added:compute bg score ====================
#     if multi_bboxes.shape[1] > 5:
#         bg_scores = multi_scores[:, -1]
#         bg_scores_15 = [bg_scores[i].repeat(int(multi_bboxes.shape[1]/5)) for i in range(bg_scores.size(0))]  # bg_score分配给每个bbox(*15)    # multi_bboxes.shape[1]/5 = 15(DOTA)  33(HRSC)
#         bg_scores_15_new = torch.Tensor(
#             [item.cpu().detach().numpy() for item in bg_scores_15])  # 由len=2000(2000,15)的tensor_list变为一个tensor
#         bg_scores = bg_scores_15_new.view(-1, 1)                    # torch.Size([30000, 1])   (30000=2000*15)
#     else:
#         bg_scores = multi_scores[:, -1]
#     # ===================== added:compute entropy =====================\
#     if multi_bboxes.shape[1] > 5:
#         probs = scores
#         log_probs = torch.log(probs)
#         entropys = (probs * log_probs).sum(1)    # 熵的负值
#         entropys_15 = [entropys[i].repeat(int(multi_bboxes.shape[1]/5)) for i in range(entropys.size(0))]  # entropy分配给每个bbox(*15)
#         entropys_15_new = torch.Tensor(
#             [item.cpu().detach().numpy() for item in entropys_15])  # 由len=2000(2000,15)的tensor_list变为一个tensor
#         entropys = entropys_15_new.view(-1, 1)                      # torch.Size([30000, 1])   (30000=2000*15)
#     else:
#         probs = scores
#         log_probs = torch.log(probs)
#         entropys = (probs * log_probs).sum(1)  # 熵的负值
#
#
#     # ==================================================================
#     labels = torch.arange(num_classes, dtype=torch.long)
#     labels = labels.view(1, -1).expand_as(scores)
#     bboxes = bboxes.reshape(-1, 5)
#     scores = scores.reshape(-1)
#     labels = labels.reshape(-1)
#
#     # ===================== added:cal iou =============================
#     # if multi_bboxes.shape[1] > 5:
#     #     stage1_rois_15 = [stage1_rois[i].repeat(15) for i in range(stage1_rois.size(0))]  # stage1_rois分配给每个bbox(*15)
#     #     stage1_rois_15_new = torch.Tensor(  # torch.Size([2000, 75])
#     #         [item.cpu().detach().numpy() for item in stage1_rois_15])
#     #     stage1_rois = stage1_rois_15_new.view(-1, 5)
#     #     bboxes_zjw = bboxes.cpu()
#     #     zjw_ious = box_iou_rotated(stage1_rois, bboxes_zjw, 'iou', True)
#     #     zjw_ious = zjw_ious.unsqueeze(-1).cpu()
#     # else:
#     # ===================== added:compute bg score ====================
#
#     # remove low scoring boxes
#     valid_mask = scores > score_thr
#     if score_factors is not None:
#         # expand the shape to match original shape of score
#         score_factors = score_factors.view(-1, 1).expand(
#             multi_scores.size(0), num_classes)
#         score_factors = score_factors.reshape(-1)
#         scores = scores * score_factors
#
#     inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
#     bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
#     entropys = entropys[inds]    # added
#     bg_scores = bg_scores[inds]  # added
#     # zjw_ious = zjw_ious[inds]    # added
#
#     if bboxes.numel() == 0:
#         dets = torch.cat([bboxes, scores[:, None]], -1)
#         if return_inds:
#             return dets, labels, inds
#         else:
#             return dets, labels, entropys, bg_scores
#
#     max_coordinate = bboxes.max()
#     offsets = labels.to(bboxes) * (max_coordinate + 1)
#     if bboxes.size(-1) == 5:
#         bboxes_for_nms = bboxes.clone()
#         bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
#     else:
#         bboxes_for_nms = bboxes + offsets[:, None]
#     _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)
#
#     if max_num > 0:
#         keep = keep[:max_num]
#
#     bboxes = bboxes[keep]
#     scores = scores[keep]
#     labels = labels[keep]
#     entropys = entropys[keep]    # added
#     bg_scores = bg_scores[keep]  # added
#     # zjw_ious = zjw_ious[keep]    # added
#
#     if return_inds:
#         return torch.cat([bboxes, scores[:, None]], 1), labels, keep
#     else:
#         return torch.cat([bboxes, scores[:, None]], 1), labels, entropys, bg_scores
#
#
# def aug_multiclass_nms_rotated(merged_bboxes, merged_labels, score_thr, nms,
#                                max_num, classes):
#     """NMS for aug multi-class bboxes.
#
#     Args:
#         multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
#         multi_scores (torch.Tensor): shape (n, #class), where the last column
#             contains scores of the background class, but this will be ignored.
#         score_thr (float): bbox threshold, bboxes with scores lower than it
#             will not be considered.
#         nms (float): Config of NMS.
#         max_num (int, optional): if there are more than max_num bboxes after
#             NMS, only top max_num will be kept. Default to -1.
#         classes (int): number of classes.
#
#     Returns:
#         tuple (dets, labels): tensors of shape (k, 5), and (k). Dets are boxes
#             with scores. Labels are 0-based.
#     """
#     bboxes, labels = [], []
#
#     for cls in range(classes):
#         cls_bboxes = merged_bboxes[merged_labels == cls]
#         inds = cls_bboxes[:, -1] > score_thr
#         if len(inds) == 0:
#             continue
#         cur_bboxes = cls_bboxes[inds, :]
#         cls_dets, _ = nms_rotated(cur_bboxes[:, :5], cur_bboxes[:, -1],
#                                   nms.iou_thr)
#         cls_labels = merged_bboxes.new_full((cls_dets.shape[0], ),
#                                             cls,
#                                             dtype=torch.long)
#         if cls_dets.size()[0] == 0:
#             continue
#         bboxes.append(cls_dets)
#         labels.append(cls_labels)
#
#     if bboxes:
#         bboxes = torch.cat(bboxes)
#         labels = torch.cat(labels)
#         if bboxes.shape[0] > max_num:
#             _, _inds = bboxes[:, -1].sort(descending=True)
#             _inds = _inds[:max_num]
#             bboxes = bboxes[_inds]
#             labels = labels[_inds]
#     else:
#         bboxes = merged_bboxes.new_zeros((0, merged_bboxes.size(-1)))
#         labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)
#
#     return bboxes, labels
