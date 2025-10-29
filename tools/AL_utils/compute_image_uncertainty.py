from tools.AL_utils.compute_budget import *
from mmdet.utils import get_root_logger
from tools.AL_utils.compute_budget import get_gt_bboxes, annotate_budget_5000_class  # 导入新的函数
from mmrotate.core.bbox.iou_calculators.rotate_iou2d_calculator import RBboxOverlaps2D
from configs._base_.datasets.dotav1 import data_root
from mmrotate.datasets.data import CLASSES
import torch.nn.functional as F
from sklearn.decomposition import PCA
import torch
import numpy as np
import os



def filter_overlapping_boxes(det_bbox, img_name, work_dir, iou_thresh=0):
    """筛选与已查询/未查询框重叠的候选框"""
    det_bbox_tensor = det_bbox[:5] if isinstance(det_bbox, torch.Tensor) else torch.tensor(det_bbox[:5],
                                                                                           device=det_bbox.device if hasattr(
                                                                                               det_bbox,
                                                                                               'device') else None)

    # 1. 已查询框重叠检查
    gt_bboxes_queried, _, _, _ = get_gt_bboxes(img_name, 'queried', work_dir + 'annfile/')
    if gt_bboxes_queried is not None and len(gt_bboxes_queried) > 0:
        gt_bboxes_queried = gt_bboxes_queried.to(det_bbox_tensor.device)
        max_iou_queried, _ = calc_iou(det_bbox_tensor, gt_bboxes_queried)
        if max_iou_queried >= 0.5:
            return False, 1

    # 2. 未查询框重叠检查
    gt_bboxes_unqueried, _, _, _ = get_gt_bboxes(img_name, 'unqueried', work_dir + 'annfile/')
    if gt_bboxes_unqueried is not None and len(gt_bboxes_unqueried) > 0:
        gt_bboxes_unqueried = gt_bboxes_unqueried.to(det_bbox_tensor.device)
        max_iou_unqueried, _ = calc_iou(det_bbox_tensor, gt_bboxes_unqueried)
        if max_iou_unqueried <= iou_thresh:
            return False, 2

    return True, 0


def ssl_select(X_U, X_L, budget, iou_thresh, data_loader, model, cfg, label_num, class_num,
               score_thresh=0.1, similarity_threshold=0.8):
    model.eval()
    model.cuda(cfg.gpu_ids[0])

    # 显存优化
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    features = []  # 用于存储提取的特征
    num_boxes_per_image = []  # 记录每个图像生成的候选框数
    feature_valid_indices = []  # 记录有效特征对应的原始图像索引
    remaining_candidates = []


    pca = PCA(n_components=128)  # 可以调整 n_components 的值来控制降维后的维度
    print(f"相似度阈值: {similarity_threshold}")

    def hook(module, input, output):
        # 这里的 output 是四维特征图 [N, C, H, W]
        # print(f"Output shape: {output.shape}")
        pooled = F.adaptive_avg_pool2d(output, (1, 1))  # 池化到 [N, C, 1, 1]
        # features.append(pooled.squeeze(-1).squeeze(-1))  # 转为 [N, C]
        current_features = pooled.squeeze(-1).squeeze(-1)  # 转为 [N, C]
        current_features_np = current_features.cpu().numpy()
        reduced_features = pca.fit_transform(current_features_np)
        # print(f"Reduced features shape: {reduced_features.shape}")
        reduced_features = torch.from_numpy(reduced_features).cuda(cfg.gpu_ids[0])
        features.append(reduced_features)

    # 注册到 bbox_roi_extractor 层
    roi_extractor_layer = model.roi_head.bbox_roi_extractor[1]
    handle = roi_extractor_layer.register_forward_hook(hook)

    all_bbox_info_list = []
    partial_queried_img = []
    feature_cache = {}

    total_candidates = 0
    filtered_stats = {'score': 0, 'overlap': [0, 0], 'diversity': 0}
    total_num = 0

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if i in set(X_U).union(set(X_L)):
                idx = np.where(i == np.array(list(set(X_U).union(set(X_L)))))[0][0]
                (shotname, ext) = os.path.splitext(data['img_metas'][0].data[0][0]['ori_filename'])
                partial_queried_img.append(shotname)

                data['img'][0].data[0] = data['img'][0].data[0].cuda(cfg.gpu_ids[0])
                data.update({'img': data['img'][0].data})
                det_bboxes, det_labels, entropys, _, _ = model(return_loss=False, return_entropy=True, **data)

                total_num += len(det_bboxes)
                valid_mask_thr = det_bboxes[:, -1] > score_thresh
                inds_thr = valid_mask_thr.nonzero(as_tuple=False).squeeze(1)
                det_bboxes_thr = det_bboxes[inds_thr]
                det_labels = det_labels[inds_thr]
                entropys = entropys[inds_thr]
                filtered_stats['score'] += len(det_bboxes) - len(det_bboxes_thr)

                if len(entropys) > 0:
                    score = det_bboxes_thr[:, -1]
                    mean_score = score.mean()
                    weight_score = 1 - mean_score
                    score = weight_score * entropys.cuda(cfg.gpu_ids[0])
                    det_labels_2D = det_labels.unsqueeze(1).cuda(cfg.gpu_ids[0])
                    image_id_2D = torch.Tensor([idx]).repeat(len(det_bboxes_thr), 1).cuda(cfg.gpu_ids[0])
                    bbox_info = torch.cat((det_bboxes_thr, det_labels_2D, score, image_id_2D), 1)
                    all_bbox_info_list.append(bbox_info)  # (cx,cy,w,h,a,score,label,entropy,image_idx)
                    num_boxes = len(bbox_info)
                    num_boxes_per_image.append(num_boxes)  # 记录候选框数目
                    feature_valid_indices.append(i)  # 记录有效特征对应的图像索引
                    total_candidates += len(det_bboxes)
                    remaining_candidates.extend(bbox_info)

        # 及时释放内存
        del data, det_bboxes, det_labels, entropys
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if i % 500 == 0:
            print(f'------ {i}/{len(data_loader.dataset)} ------')

    # 特征索引映射
    index_mapping = []
    for idx, (img_idx, num_box) in enumerate(zip(feature_valid_indices, num_boxes_per_image)):
        index_mapping.extend([(img_idx, box_idx) for box_idx in range(num_box)])

    # 排序与预算分配
    all_bbox_info = torch.cat(all_bbox_info_list)[torch.cat(all_bbox_info_list)[:, -2].argsort()]
    label_weights = F.softmax(1 - label_num / label_num.sum(), dim=0)
    class_budgets = ((0.5 * label_weights + 0.5 * (1 / class_num)) * budget)
    total_budget = budget

    # 第二阶段：筛选与标注
    selected_indices = []
    feature_cache = {}

    # 新增：存储因类别预算不足被跳过的候选框
    budget_exhausted_candidates = []

    for idx in range(len(all_bbox_info)):
        if total_budget <= 0:
            break

        candidate = all_bbox_info[idx]
        img_idx = int(candidate[-1].item())
        img_name = partial_queried_img[img_idx]
        det_bbox = candidate[:6]
        class_id = int(candidate[6].item())

        # 类别预算检查（严格模式）
        if class_budgets[class_id] <= 0:
            budget_exhausted_candidates.append((idx, candidate))  # 记录候选框及索引
            continue  # 跳过，进入剩余处理阶段

        # 第一次重叠过滤
        pass_overlap, reason = filter_overlapping_boxes(det_bbox, img_name, cfg.work_dir[:-6], iou_thresh)
        if not pass_overlap:
            filtered_stats['overlap'][reason - 1] += 1
            continue

        # 预算检查
        if class_budgets[class_id] <= 0 or total_budget <= 0:
            continue

        # 多样性检查
        try:
            img_idx_map, box_idx = index_mapping[idx]
            current_feature = features[img_idx_map][box_idx].unsqueeze(0)
        except IndexError:
            continue

        if selected_indices:
            selected_features = torch.stack([feature_cache[i] for i in selected_indices])
            similarity = F.cosine_similarity(current_feature, selected_features).max()
            if similarity > similarity_threshold:
                filtered_stats['diversity'] += 1
                continue

        # 更新标注
        gt_bboxes, _, _, gt_raw = get_gt_bboxes(img_name, 'unqueried', cfg.work_dir[:-6] + 'annfile/')
        if gt_bboxes is not None and len(gt_bboxes) > 0:
            # 确保 gt_bboxes 和 det_bbox 在同一设备上
            device = gt_bboxes.device
            det_bbox = det_bbox.to(device)
            max_iou, ious = calc_iou(det_bbox[:5], gt_bboxes)
            if max_iou > iou_thresh:
                # 移动匹配的GT框
                queried_gt = gt_raw[(ious == max_iou).cpu().detach().numpy()]
                unqueried_gt = gt_raw[(ious < max_iou).cpu().detach().numpy()]
                if len(queried_gt) > 0:
                    class_budgets[class_id] -= 1
                    total_budget -= 1
                    selected_indices.append(idx)
                    feature_cache[idx] = current_feature.squeeze(0)
                    updata_image_info(queried_gt, unqueried_gt, img_name, cfg.work_dir[:-6] + 'annfile/')

    if total_budget > 0:
        for idx, candidate in budget_exhausted_candidates:
            if total_budget <= 0:
                break

            img_idx = int(candidate[-1].item())
            img_name = partial_queried_img[img_idx]
            det_bbox = candidate[:6]
            class_id = int(candidate[6].item())

            pass_overlap, reason = filter_overlapping_boxes(det_bbox, img_name, cfg.work_dir[:-6], iou_thresh)
            if not pass_overlap:
                filtered_stats['overlap'][reason - 1] += 1
                continue

            try:
                img_idx_map, box_idx = index_mapping[idx]
                current_feature = features[img_idx_map][box_idx].unsqueeze(0)
            except IndexError:
                continue

            if selected_indices:
                selected_features = torch.stack([feature_cache[i] for i in selected_indices])
                similarity = F.cosine_similarity(current_feature, selected_features).max()
                if similarity > similarity_threshold:
                    filtered_stats['diversity'] += 1
                    continue

            gt_bboxes, _, _, gt_raw = get_gt_bboxes(img_name, 'unqueried', cfg.work_dir[:-6] + 'annfile/')
            if gt_bboxes is not None and len(gt_bboxes) > 0:
                device = det_bbox.device
                gt_bboxes = gt_bboxes.to(device)
                max_iou, ious = calc_iou(det_bbox[:5], gt_bboxes)
                if max_iou > iou_thresh:
                    queried_gt = gt_raw[(ious == max_iou).cpu().detach().numpy()]
                    unqueried_gt = gt_raw[(ious < max_iou).cpu().detach().numpy()]
                    if len(queried_gt) > 0:
                        total_budget -= 1
                        selected_indices.append(idx)
                        feature_cache[idx] = current_feature.squeeze(0)
                        updata_image_info(queried_gt, unqueried_gt, img_name, cfg.work_dir[:-6] + 'annfile/')

    # 最终文件整理
    form_file(cfg.work_dir[:-6] + 'annfile/')
    handle.remove()

    # 打印统计信息
    print(f"总候选: {total_num} 分数过滤: {filtered_stats['score']}")
    print(f"重叠过滤(已查询/未查询): {filtered_stats['overlap']}")
    print(f"多样性过滤: {filtered_stats['diversity']}")
    print(f"相似度阈值: {similarity_threshold}")

    return selected_indices


def usgss_select(X_U, X_L, budget, iou_thresh, data_loader, model, cfg, label_num, class_num,
                 score_thresh=0.1, similarity_threshold=0.8, aspect_angle_threshold=0.95):
    model.eval()
    model.cuda(cfg.gpu_ids[0])

    global high_count_categories
    folder_path = data_root + 'trainval/annfiles'
    high_count_categories = count_categories_in_folder(folder_path)

    # 显存优化配置
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # 初始化数据存储
    num_boxes_per_image = []
    feature_valid_indices = []
    all_bbox_info_list = []
    partial_queried_img = []
    index_mapping = []

    # PCA降维初始化
    pca = PCA(n_components=128)
    print(f"长宽比/角度相似度阈值: {aspect_angle_threshold}")

    # 数据预处理循环
    total_candidates = 0
    total_num = 0
    filtered_stats = {'score': 0, 'overlap': [0, 0], 'aspect_angle': 0}

    for i, data in enumerate(data_loader):
        if i in set(X_U).union(set(X_L)):
            idx = np.where(i == np.array(list(set(X_U).union(set(X_L)))))[0][0]
            shotname = os.path.splitext(data['img_metas'][0].data[0][0]['ori_filename'])[0]
            partial_queried_img.append(shotname)

            # 前向传播
            with torch.no_grad():
                data['img'][0].data[0] = data['img'][0].data[0].cuda(cfg.gpu_ids[0])
                data.update({'img': data['img'][0].data})
                det_bboxes, det_labels, entropys, _, _ = model(return_loss=False, return_entropy=True, **data)

            # 分数过滤
            total_num += len(det_bboxes)
            valid_mask = det_bboxes[:, -1] > score_thresh
            det_bboxes_thr = det_bboxes[valid_mask]
            det_labels = det_labels[valid_mask]
            entropys = entropys[valid_mask]
            filtered_stats['score'] += len(det_bboxes) - len(det_bboxes_thr)

            if len(det_bboxes_thr) > 0:
                # 构建候选信息
                scores = det_bboxes_thr[:, -1]
                mean_score = scores.mean()
                weight_score = 1 - mean_score
                score = weight_score * entropys.cuda(cfg.gpu_ids[0])
                # score = entropys.cuda(cfg.gpu_ids[0])
                score = score.unsqueeze(1)  # Add an extra dimension
                det_labels_2D = det_labels.unsqueeze(1).cuda(cfg.gpu_ids[0])
                image_id_2D = torch.tensor([idx]).repeat(len(det_bboxes_thr), 1).cuda(cfg.gpu_ids[0])
                # 打印det_bboxes_thr, det_labels_2D, score, image_id_2D的形状
                # print(f"det_bboxes_thr shape: {det_bboxes_thr.shape}")
                # print(f"det_labels_2D shape: {det_labels_2D.shape}")
                # print(f"score shape: {score.shape}")
                # print(f"image_id_2D shape: {image_id_2D.shape}")
                bbox_info = torch.cat((det_bboxes_thr, det_labels_2D, score, image_id_2D), 1)

                all_bbox_info_list.append(bbox_info)
                num_boxes = len(bbox_info)
                num_boxes_per_image.append(num_boxes)
                feature_valid_indices.append(i)
                total_candidates += len(bbox_info)
                index_mapping.extend([(i, box_idx) for box_idx in range(num_boxes)])

            del data, det_bboxes, det_labels, entropys, valid_mask
            torch.cuda.empty_cache()

        if i % 500 == 0:
            print(f'Processing {i}/{len(data_loader.dataset)}')

    # 合并所有候选信息
    if not all_bbox_info_list:
        return []

    all_bbox_info = torch.cat(all_bbox_info_list)
    del all_bbox_info_list
    torch.cuda.empty_cache()

    all_bbox_info = all_bbox_info[all_bbox_info[:, -2].argsort()]  # 按加权分数排序

    # 向量化预计算长宽比和角度
    def precompute_aspect_angles(bboxes):
        """批量计算长宽比和角度"""
        # 假设bboxes格式为 [cx, cy, w, h, angle, score, class, ...]
        w = bboxes[:, 2]
        h = bboxes[:, 3]
        angles = bboxes[:, 4]

        ars = torch.max(w, h) / torch.min(w, h)
        ars[torch.isinf(ars)] = 0.0  # 处理可能的除零错误

        angles_deg = torch.rad2deg(angles) % 180.0

        return ars.cuda(cfg.gpu_ids[0]), angles_deg.cuda(cfg.gpu_ids[0])

    all_ars, all_angles = precompute_aspect_angles(all_bbox_info)

    # 初始化类别预算
    label_weights = torch.nn.functional.softmax(1 - label_num / label_num.sum(), dim=0)
    class_budgets = ((0.5 * label_weights + 0.5 * (1 / class_num)) * budget).to(torch.int32)
    total_budget = budget

    # 维护已选框数据
    selected_indices = []
    feature_cache = {}
    class_selected_data = {
        cid: {'ars': torch.tensor([], device='cuda'),
              'angles': torch.tensor([], device='cuda')}
        for cid in range(class_num)
    }
    # 新增：存储因类别预算不足被跳过的候选框
    budget_exhausted_candidates = []

    # # 向量化相似度计算
    def calculate_batch_similarity(current_ar, current_angle, class_ars, class_angles):
        """GPU加速的批量相似度计算"""
        # 计算长宽比相似度 0.3 5
        ar_diff = (current_ar - class_ars).pow(2)
        sim_ar = torch.exp(-ar_diff / (2 * 0.3 ** 2))

        # 计算角度相似度（考虑周期性）
        angle_diff = torch.abs(current_angle - class_angles)
        angle_diff = torch.min(angle_diff, 180 - angle_diff)
        sim_angle = torch.exp(-angle_diff.pow(2) / (2 * 5 ** 2))

        # 组合相似度
        return 0.5 * sim_ar + 0.5 * sim_angle

    # 主选择循环
    for idx in range(len(all_bbox_info)):
        if total_budget <= 0:
            break

        candidate = all_bbox_info[idx]
        img_idx = int(candidate[-1].item())
        img_name = partial_queried_img[img_idx]
        det_bbox = candidate[:6]
        current_class = int(candidate[6].item())
        current_ar = all_ars[idx]
        current_angle = all_angles[idx]

        # 预算检查
        if class_budgets[current_class] <= 0:
            budget_exhausted_candidates.append((idx, candidate))  # 记录候选框及索引
            continue

        # 重叠过滤
        pass_overlap, reason = filter_overlapping_boxes(det_bbox, img_name, cfg.work_dir[:-6], iou_thresh)
        if not pass_overlap:
            filtered_stats['overlap'][reason - 1] += 1
            continue

        # 长宽比/角度相似度检查（向量化版本）
        class_data = class_selected_data[current_class]
        if len(class_data['ars']) > 0 and current_class in high_count_categories:
            similarities = calculate_batch_similarity(
                current_ar, current_angle,
                class_data['ars'], class_data['angles']
                # , high_count_categories, current_class
            )
            # print(f"相似度1: {similarities}")
            max_similarity = similarities.max()
            # print(f"最大相似度: {max_similarity}")
            del similarities  # 立即释放临时张量
            if max_similarity > aspect_angle_threshold:
                filtered_stats['aspect_angle'] += 1
                continue

        # 标注更新
        gt_bboxes, _, _, gt_raw = get_gt_bboxes(img_name, 'unqueried', cfg.work_dir[:-6] + 'annfile/')
        if gt_bboxes is not None and len(gt_bboxes) > 0:
            device = gt_bboxes.device
            det_bbox = det_bbox.to(device)
            max_iou, ious = calc_iou(det_bbox[:5], gt_bboxes)
            if max_iou > iou_thresh:
                queried_gt = gt_raw[(ious == max_iou).cpu().detach().numpy()]
                unqueried_gt = gt_raw[(ious < max_iou).cpu().detach().numpy()]
                if len(queried_gt) > 0:
                    class_budgets[current_class] -= 1
                    total_budget -= 1
                    selected_indices.append(idx)
                    # feature_cache[idx] = current_feature.squeeze(0)

                    # 更新类别数据
                    class_selected_data[current_class]['ars'] = torch.cat([
                        class_selected_data[current_class]['ars'],
                        current_ar.unsqueeze(0)
                    ])
                    class_selected_data[current_class]['angles'] = torch.cat([
                        class_selected_data[current_class]['angles'],
                        current_angle.unsqueeze(0)
                    ])

                    updata_image_info(queried_gt, unqueried_gt, img_name, cfg.work_dir[:-6] + 'annfile/')
                    del queried_gt, unqueried_gt  # 立即释放临时张量

        # 仅在不再需要这些变量时释放
        del candidate, img_idx, img_name, current_class

    # 处理预算耗尽的候选框（仅在此阶段跨类别补充）
    if total_budget > 0:
        for idx, candidate in budget_exhausted_candidates:
            if total_budget <= 0:
                break

            img_idx = int(candidate[-1].item())
            img_name = partial_queried_img[img_idx]
            det_bbox = candidate[:6]
            current_class = int(candidate[6].item())
            current_ar = all_ars[idx]
            current_angle = all_angles[idx]

            # 重叠过滤
            pass_overlap, reason = filter_overlapping_boxes(det_bbox, img_name, cfg.work_dir[:-6], iou_thresh)
            if not pass_overlap:
                filtered_stats['overlap'][reason - 1] += 1
                continue

            # 长宽比/角度相似度检查
            class_data = class_selected_data[current_class]
            if len(class_data['ars']) > 0 and current_class in high_count_categories:
                similarities = calculate_batch_similarity(
                    current_ar, current_angle,
                    class_data['ars'], class_data['angles']
                    # , high_count_categories, current_class
                )
                # print(f"相似度2: {similarities}")
                max_similarity = similarities.max()
                # print(f"最大相似度: {max_similarity}")
                del similarities  # 立即释放临时张量
                if max_similarity > aspect_angle_threshold:
                    filtered_stats['aspect_angle'] += 1
                    continue

            # 标注更新
            gt_bboxes, _, _, gt_raw = get_gt_bboxes(img_name, 'unqueried', cfg.work_dir[:-6] + 'annfile/')
            if gt_bboxes is not None and len(gt_bboxes) > 0:
                device = det_bbox.device
                gt_bboxes = gt_bboxes.to(device)
                max_iou, ious = calc_iou(det_bbox[:5], gt_bboxes)
                if max_iou > iou_thresh:
                    queried_gt = gt_raw[(ious == max_iou).cpu().detach().numpy()]
                    unqueried_gt = gt_raw[(ious < max_iou).cpu().detach().numpy()]
                    if len(queried_gt) > 0:
                        total_budget -= 1
                        selected_indices.append(idx)
                        # feature_cache[idx] = current_feature.squeeze(0)

                        # 更新类别数据
                        class_selected_data[current_class]['ars'] = torch.cat([
                            class_selected_data[current_class]['ars'],
                            current_ar.unsqueeze(0)
                        ])
                        class_selected_data[current_class]['angles'] = torch.cat([
                            class_selected_data[current_class]['angles'],
                            current_angle.unsqueeze(0)
                        ])

                        updata_image_info(queried_gt, unqueried_gt, img_name, cfg.work_dir[:-6] + 'annfile/')
                        del queried_gt, unqueried_gt  # 立即释放临时张量

            # 仅在不再需要这些变量时释放
            del candidate, img_idx, img_name, current_class

    # 收尾工作
    form_file(cfg.work_dir[:-6] + 'annfile/')

    # 释放不再使用的大型数据结构
    del all_bbox_info, all_ars, all_angles, class_selected_data
    torch.cuda.empty_cache()

    print(f"总候选: {total_num} 过滤统计:")
    print(f"分数过滤: {filtered_stats['score']}")
    print(f"重叠过滤(已查询/未查询): {filtered_stats['overlap']}")
    print(f"长宽比/角度过滤: {filtered_stats['aspect_angle']}")

    return selected_indices


def count_categories_in_folder(folder_path):
    category_counts = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            category_name = parts[-2]  # 假设最后第二个元素是类别名称
                            # 将类别名称映射为索引
                            if category_name in CLASSES:
                                category_idx = CLASSES.index(category_name)
                                category_counts[category_idx] = category_counts.get(category_idx, 0) + 1

    # 确保所有类别都在结果中，数量为0的类别也会显示
    complete_counts = {idx: category_counts.get(idx, 0) for idx in range(len(CLASSES))}

    # 筛选出数量大于1500的类别（返回索引列表）
    high_count_indices = [idx for idx, count in complete_counts.items() if
                          count >= 1200]  # dotav1 - 1500, dior_r - 1200
    return high_count_indices


def udc_select(X_U, X_L, budget, iou_thresh, data_loader, model, cfg, label_num, class_num,
               score_thresh=0.1, similarity_threshold=0.8, aspect_angle_threshold=0.95):
    model.eval()
    model.cuda(cfg.gpu_ids[0])

    folder_path = data_root + 'trainval/annfiles'
    high_count_categories = count_categories_in_folder(folder_path)
    # print(f"high_count_categories: {high_count_categories, type(high_count_categories)}")

    # 显存优化配置
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # 初始化数据存储
    features = []
    num_boxes_per_image = []
    feature_valid_indices = []
    all_bbox_info_list = []
    partial_queried_img = []
    index_mapping = []

    # PCA降维初始化
    pca = PCA(n_components=128)
    print(f"特征相似度阈值: {similarity_threshold}, 长宽比/角度相似度阈值: {aspect_angle_threshold}")

    # 注册特征提取钩子
    def hook(module, input, output):
        pooled = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
        current_features = pooled.squeeze(-1).squeeze(-1)
        current_features_np = current_features.detach().cpu().numpy()
        reduced_features = pca.fit_transform(current_features_np)
        features.append(torch.from_numpy(reduced_features).cuda(cfg.gpu_ids[0]))

    roi_extractor_layer = model.roi_head.bbox_roi_extractor[1]
    handle = roi_extractor_layer.register_forward_hook(hook)

    # 数据预处理循环
    total_candidates = 0
    total_num = 0
    filtered_stats = {'score': 0, 'overlap': [0, 0], 'diversity': 0, 'aspect_angle': 0}

    for i, data in enumerate(data_loader):
        if i in set(X_U).union(set(X_L)):
            idx = np.where(i == np.array(list(set(X_U).union(set(X_L)))))[0][0]
            shotname = os.path.splitext(data['img_metas'][0].data[0][0]['ori_filename'])[0]
            partial_queried_img.append(shotname)

            # 前向传播
            with torch.no_grad():
                data['img'][0].data[0] = data['img'][0].data[0].cuda(cfg.gpu_ids[0])
                data.update({'img': data['img'][0].data})
                det_bboxes, det_labels, entropys, _, _ = model(return_loss=False, return_entropy=True, **data)

            # 分数过滤
            total_num += len(det_bboxes)
            valid_mask = det_bboxes[:, -1] > score_thresh
            det_bboxes_thr = det_bboxes[valid_mask]
            det_labels = det_labels[valid_mask]
            entropys = entropys[valid_mask]
            filtered_stats['score'] += len(det_bboxes) - len(det_bboxes_thr)

            if len(det_bboxes_thr) > 0:
                # 构建候选信息
                scores = det_bboxes_thr[:, -1]
                mean_score = scores.mean()
                weight_score = 1 - mean_score
                score = weight_score * entropys.cuda(cfg.gpu_ids[0])
                # score = entropys.cuda(cfg.gpu_ids[0])
                score = score.unsqueeze(1)  # Add an extra dimension
                det_labels_2D = det_labels.unsqueeze(1).cuda(cfg.gpu_ids[0])
                image_id_2D = torch.tensor([idx]).repeat(len(det_bboxes_thr), 1).cuda(cfg.gpu_ids[0])
                # 打印det_bboxes_thr, det_labels_2D, score, image_id_2D的形状
                # print(f"det_bboxes_thr shape: {det_bboxes_thr.shape}")
                # print(f"det_labels_2D shape: {det_labels_2D.shape}")
                # print(f"score shape: {score.shape}")
                # print(f"image_id_2D shape: {image_id_2D.shape}")
                bbox_info = torch.cat((det_bboxes_thr, det_labels_2D, score, image_id_2D), 1)

                all_bbox_info_list.append(bbox_info)
                num_boxes = len(bbox_info)
                num_boxes_per_image.append(num_boxes)
                feature_valid_indices.append(i)
                total_candidates += len(bbox_info)
                index_mapping.extend([(i, box_idx) for box_idx in range(num_boxes)])

            del data, det_bboxes, det_labels, entropys, valid_mask
            torch.cuda.empty_cache()

        if i % 500 == 0:
            print(f'Processing {i}/{len(data_loader.dataset)}')

    # 合并所有候选信息
    if not all_bbox_info_list:
        handle.remove()
        return []

    all_bbox_info = torch.cat(all_bbox_info_list)
    del all_bbox_info_list
    torch.cuda.empty_cache()

    all_bbox_info = all_bbox_info[all_bbox_info[:, -2].argsort()]  # 按加权分数排序

    # 向量化预计算长宽比和角度
    def precompute_aspect_angles(bboxes):
        """批量计算长宽比和角度"""
        # 假设bboxes格式为 [cx, cy, w, h, angle, score, class, ...]
        w = bboxes[:, 2]
        h = bboxes[:, 3]
        angles = bboxes[:, 4]

        ars = torch.max(w, h) / torch.min(w, h)
        ars[torch.isinf(ars)] = 0.0  # 处理可能的除零错误

        angles_deg = torch.rad2deg(angles) % 180.0

        return ars.cuda(cfg.gpu_ids[0]), angles_deg.cuda(cfg.gpu_ids[0])

    all_ars, all_angles = precompute_aspect_angles(all_bbox_info)

    # 初始化类别预算
    label_weights = torch.nn.functional.softmax(1 - label_num / label_num.sum(), dim=0)
    class_budgets = ((0.5 * label_weights + 0.5 * (1 / class_num)) * budget).to(torch.int32)
    total_budget = budget

    # 维护已选框数据
    selected_indices = []
    feature_cache = {}
    class_selected_data = {
        cid: {'ars': torch.tensor([], device='cuda'),
              'angles': torch.tensor([], device='cuda')}
        for cid in range(class_num)
    }
    # 新增：存储因类别预算不足被跳过的候选框
    budget_exhausted_candidates = []

    # # 向量化相似度计算
    def calculate_batch_similarity(current_ar, current_angle, class_ars, class_angles):
        """GPU加速的批量相似度计算"""
        # 计算长宽比相似度 0.3 5
        ar_diff = (current_ar - class_ars).pow(2)
        sim_ar = torch.exp(-ar_diff / (2 * 0.3 ** 2))

        # 计算角度相似度（考虑周期性）
        angle_diff = torch.abs(current_angle - class_angles)
        angle_diff = torch.min(angle_diff, 180 - angle_diff)
        sim_angle = torch.exp(-angle_diff.pow(2) / (2 * 5 ** 2))

        # 组合相似度
        return 0.5 * sim_ar + 0.5 * sim_angle

    # 主选择循环
    for idx in range(len(all_bbox_info)):
        if total_budget <= 0:
            break

        candidate = all_bbox_info[idx]
        img_idx = int(candidate[-1].item())
        img_name = partial_queried_img[img_idx]
        det_bbox = candidate[:6]
        current_class = int(candidate[6].item())
        current_ar = all_ars[idx]
        current_angle = all_angles[idx]

        # 预算检查
        if class_budgets[current_class] <= 0:
            budget_exhausted_candidates.append((idx, candidate))  # 记录候选框及索引
            continue

        # 重叠过滤
        pass_overlap, reason = filter_overlapping_boxes(det_bbox, img_name, cfg.work_dir[:-6], iou_thresh)
        if not pass_overlap:
            filtered_stats['overlap'][reason - 1] += 1
            continue

        # 特征相似度检查
        try:
            img_idx_map, box_idx = index_mapping[idx]
            current_feature = features[img_idx_map][box_idx].unsqueeze(0)
        except (IndexError, KeyError):
            continue

        if selected_indices:
            selected_features = torch.stack([feature_cache[i] for i in selected_indices])
            similarity = torch.nn.functional.cosine_similarity(current_feature, selected_features).max()
            del selected_features  # 立即释放临时张量
            if similarity > similarity_threshold:
                filtered_stats['diversity'] += 1
                continue

        # 长宽比/角度相似度检查（向量化版本）
        class_data = class_selected_data[current_class]
        # print(f"current_class: {current_class}")
        if len(class_data['ars']) > 0 and current_class in high_count_categories:
            # if len(class_data['ars']) > 0:
            # print(f"current_class: {current_class}")
            similarities = calculate_batch_similarity(
                current_ar, current_angle,
                class_data['ars'], class_data['angles']
            )
            # print(f"相似度1: {similarities}")
            max_similarity = similarities.max()
            # print(f"最大相似度: {max_similarity}")
            del similarities  # 立即释放临时张量
            if max_similarity > aspect_angle_threshold:
                filtered_stats['aspect_angle'] += 1
                continue

        # 标注更新
        gt_bboxes, _, _, gt_raw = get_gt_bboxes(img_name, 'unqueried', cfg.work_dir[:-6] + 'annfile/')
        if gt_bboxes is not None and len(gt_bboxes) > 0:
            device = gt_bboxes.device
            det_bbox = det_bbox.to(device)
            max_iou, ious = calc_iou(det_bbox[:5], gt_bboxes)
            if max_iou > iou_thresh:
                queried_gt = gt_raw[(ious == max_iou).cpu().detach().numpy()]
                unqueried_gt = gt_raw[(ious < max_iou).cpu().detach().numpy()]
                if len(queried_gt) > 0:
                    class_budgets[current_class] -= 1
                    total_budget -= 1
                    selected_indices.append(idx)
                    feature_cache[idx] = current_feature.squeeze(0)

                    # 更新类别数据
                    class_selected_data[current_class]['ars'] = torch.cat([
                        class_selected_data[current_class]['ars'],
                        current_ar.unsqueeze(0)
                    ])
                    class_selected_data[current_class]['angles'] = torch.cat([
                        class_selected_data[current_class]['angles'],
                        current_angle.unsqueeze(0)
                    ])

                    updata_image_info(queried_gt, unqueried_gt, img_name, cfg.work_dir[:-6] + 'annfile/')
                    del queried_gt, unqueried_gt  # 立即释放临时张量

        # 仅在不再需要这些变量时释放
        del candidate, img_idx, img_name, current_class

    # 处理预算耗尽的候选框（仅在此阶段跨类别补充）
    if total_budget > 0:
        for idx, candidate in budget_exhausted_candidates:
            if total_budget <= 0:
                break

            img_idx = int(candidate[-1].item())
            img_name = partial_queried_img[img_idx]
            det_bbox = candidate[:6]
            current_class = int(candidate[6].item())
            current_ar = all_ars[idx]
            current_angle = all_angles[idx]

            # 重叠过滤
            pass_overlap, reason = filter_overlapping_boxes(det_bbox, img_name, cfg.work_dir[:-6], iou_thresh)
            if not pass_overlap:
                filtered_stats['overlap'][reason - 1] += 1
                continue

            # 特征相似度检查
            try:
                img_idx_map, box_idx = index_mapping[idx]
                current_feature = features[img_idx_map][box_idx].unsqueeze(0)
            except (IndexError, KeyError) as e:
                print(f"索引错误: {e}")
                continue

            if selected_indices:
                selected_features = torch.stack([feature_cache[i] for i in selected_indices])
                similarity = torch.nn.functional.cosine_similarity(current_feature, selected_features).max()
                del selected_features  # 立即释放临时张量
                if similarity > similarity_threshold:
                    filtered_stats['diversity'] += 1
                    continue

            # 长宽比/角度相似度检查
            class_data = class_selected_data[current_class]
            if len(class_data['ars']) > 0 and current_class in high_count_categories:
                # if len(class_data['ars']) > 0:
                similarities = calculate_batch_similarity(
                    current_ar, current_angle,
                    class_data['ars'], class_data['angles']
                )
                # print(f"相似度2: {similarities}")
                max_similarity = similarities.max()
                # print(f"最大相似度: {max_similarity}")
                del similarities  # 立即释放临时张量
                if max_similarity > aspect_angle_threshold:
                    filtered_stats['aspect_angle'] += 1
                    continue

            # 标注更新
            gt_bboxes, _, _, gt_raw = get_gt_bboxes(img_name, 'unqueried', cfg.work_dir[:-6] + 'annfile/')
            if gt_bboxes is not None and len(gt_bboxes) > 0:
                device = det_bbox.device
                gt_bboxes = gt_bboxes.to(device)
                max_iou, ious = calc_iou(det_bbox[:5], gt_bboxes)
                if max_iou > iou_thresh:
                    queried_gt = gt_raw[(ious == max_iou).cpu().detach().numpy()]
                    unqueried_gt = gt_raw[(ious < max_iou).cpu().detach().numpy()]
                    if len(queried_gt) > 0:
                        total_budget -= 1
                        selected_indices.append(idx)
                        feature_cache[idx] = current_feature.squeeze(0)

                        # 更新类别数据
                        class_selected_data[current_class]['ars'] = torch.cat([
                            class_selected_data[current_class]['ars'],
                            current_ar.unsqueeze(0)
                        ])
                        class_selected_data[current_class]['angles'] = torch.cat([
                            class_selected_data[current_class]['angles'],
                            current_angle.unsqueeze(0)
                        ])

                        updata_image_info(queried_gt, unqueried_gt, img_name, cfg.work_dir[:-6] + 'annfile/')
                        del queried_gt, unqueried_gt  # 立即释放临时张量

            # 仅在不再需要这些变量时释放
            del candidate, img_idx, img_name, current_class

    # 收尾工作
    handle.remove()
    form_file(cfg.work_dir[:-6] + 'annfile/')

    # 释放不再使用的大型数据结构
    del features, all_bbox_info, all_ars, all_angles, feature_cache, class_selected_data
    torch.cuda.empty_cache()

    print(f"总候选: {total_num} 过滤统计:")
    print(f"分数过滤: {filtered_stats['score']}")
    print(f"重叠过滤(已查询/未查询): {filtered_stats['overlap']}")
    print(f"多样性过滤: {filtered_stats['diversity']}")
    print(f"长宽比/角度过滤: {filtered_stats['aspect_angle']}")

    return selected_indices


def mus_cdb_select(X_U, budget, iou_thresh, data_loader, model, cfg, label_num, class_num, score_thresh):  # mus-cdb
    """Uncertainty in BAOD uncertainty evaluation method.
          Average entropy of each predicted box.
          or
          Mean conf of the first M boxes.
   """
    model.eval()
    model.cuda(cfg.gpu_ids[0])
    all_bbox_info = torch.Tensor().cuda(cfg.gpu_ids[0])
    partial_queried_img = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if i in X_U:
                idx = np.where(i == X_U)[0][0]
                (shotname, ext) = os.path.splitext(data['img_metas'][0].data[0][0]['ori_filename'])
                partial_queried_img.append(shotname)
                data['img'][0].data[0] = data['img'][0].data[0].cuda(cfg.gpu_ids[0])
                data.update({'img': data['img'][0].data})
                det_bboxes, det_labels, entropys, _, _ = model(return_loss=False, return_entropy=True, **data)
                valid_mask_thr = det_bboxes[:, -1] > score_thresh
                inds_thr = valid_mask_thr.nonzero(as_tuple=False).squeeze(1)
                det_bboxes_thr = det_bboxes[inds_thr]
                if len(entropys) > 0:
                    score = det_bboxes_thr[:, -1]
                    mean_score = score.mean()
                    weight_score = 1 - mean_score
                    score = weight_score * entropys.cuda(cfg.gpu_ids[0])
                    score = score.unsqueeze(1)  # Add an extra dimension
                    det_labels_2D = det_labels.unsqueeze(1).cuda(cfg.gpu_ids[0])
                    image_id_2D = torch.Tensor([idx]).repeat(len(det_bboxes), 1).cuda(cfg.gpu_ids[0])
                    bbox_info = torch.cat((det_bboxes, det_labels_2D, score, image_id_2D),
                                          1)  # (cx,cy,w,h,a,score,label,entropy,image_idx)

                    # print(f"Shape mismatch: all_bbox_info {all_bbox_info.shape}, bbox_info {bbox_info.shape}")

                    all_bbox_info = torch.cat((all_bbox_info, bbox_info), dim=0)
            if i % 500 == 0:
                print(f'------ {i}/{len(data_loader.dataset)} ------')
    index = (all_bbox_info[:, -2]).argsort()  # asend, small -> large
    new_all_bbox_info = all_bbox_info[index].cpu()
    torch.save(new_all_bbox_info,
               cfg.work_dir + '/all_candidate_bbox_info.pt')
    annotate_budget_5000_class(new_all_bbox_info, partial_queried_img, cfg.work_dir[:-6] + 'annfile/',
                               budget, iou_thresh, label_num, class_num)


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


class SelectionMethod:
    """
    Abstract base class for selection methods,
    which allow to select a subset of indices from the pool set as the next batch to label for Batch Active Learning.
    """

    def __init__(self, al_sample, X_L, X_U, data_loader, model, cfg, all_image_name):
        super().__init__()
        self.logger = get_root_logger()
        self.al_sample = al_sample
        self.X_L = X_L
        self.X_U = X_U
        self.data_loader = data_loader
        self.model = model
        self.cfg = cfg
        self.all_image_name = all_image_name

    def select(self, budget, iou_thresh, label_num=0, score_thresh=0.05, class_num=15, similarity_threshold=0.8):
        """
        Select selection_size elements from the pool set
        (which is assumed to be given in the constructor of the corresponding subclass).
        This method needs to be implemented by subclasses.
        It is assumed that this method is only called once per object, since it can modify the state of the object.

        Args:
            selection_size (int): how much images selected in one cycle

        Returns:
            idxs_selected (np.ndarray): index of chosen images
        """
        if self.al_sample == 'mus-cdb':
            self.logger.info(f'------ mus-cdb ------')
            return mus_cdb_select(self.X_U, budget, iou_thresh, self.data_loader, self.model,
                                  self.cfg, label_num, class_num, score_thresh)
        elif self.al_sample == 'ssl':
            self.logger.info(f'------ ssl ------')
            return ssl_select(self.X_U, self.X_L, budget, iou_thresh, self.data_loader, self.model,
                              self.cfg, label_num, class_num, score_thresh, similarity_threshold)
        elif self.al_sample == 'udc':
            self.logger.info(f'------ udc ------')
            return udc_select(self.X_U, self.X_L, budget, iou_thresh, self.data_loader, self.model,
                              self.cfg, label_num, class_num, score_thresh, similarity_threshold,
                              aspect_angle_threshold=0.95)
        elif self.al_sample == 'usgss':
            self.logger.info(f'------ usgss ------')
            return usgss_select(self.X_U, self.X_L, budget, iou_thresh, self.data_loader, self.model,
                                self.cfg, label_num, class_num, score_thresh, similarity_threshold,
                                aspect_angle_threshold=0.95)

