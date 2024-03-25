# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from mmengine.structures import BaseDataElement

from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import reduce_mean
from mmdet.structures.bbox import distance2bbox
                                   
EPS = 1e-5

class DDQAuxLoss(nn.Module):
    """DDQ auxiliary branches loss for dense queries.

    Args:
        loss_cls (dict):
            Configuration of classification loss function.
        loss_bbox (dict):
            Configuration of bbox regression loss function.
        train_cfg (dict):
            Configuration of gt targets assigner for each predicted bbox.
    """

    def __init__(
        self,
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(type='TopkHungarianAssigner', topk=8),
            alpha=1,
            beta=6),
    ):
        super(DDQAuxLoss, self).__init__()
        self.train_cfg = train_cfg
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])

        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = TASK_UTILS.build(sampler_cfg)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, alignment_metrics):
        """Calculate auxiliary branches loss for dense queries for one image.

        Args:
            cls_score (Tensor): Predicted normalized classification
                scores for one image, has shape (num_dense_queries,
                cls_out_channels).
            bbox_pred (Tensor): Predicted unnormalized bbox coordinates
                for one image, has shape (num_dense_queries, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
            labels (Tensor): Labels for one image.
            label_weights (Tensor): Label weights for one image.
            bbox_targets (Tensor): Bbox targets for one image.
            alignment_metrics (Tensor): Normalized alignment metrics for one
                image.

        Returns:
            tuple: A tuple of loss components and loss weights.
        """
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, alignment_metrics)
        cls_loss_func = self.loss_cls

        loss_cls = cls_loss_func(
            cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = cls_score.size(-1)
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets

            # regression loss
            pos_bbox_weight = alignment_metrics[pos_inds]

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, alignment_metrics.sum(
        ), pos_bbox_weight.sum()

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas,
             **kwargs):
        """Calculate auxiliary branches loss for dense queries.

        Args:
            cls_scores (Tensor): Predicted normalized classification
                scores, has shape (bs, num_dense_queries,
                cls_out_channels).
            bbox_preds (Tensor): Predicted unnormalized bbox coordinates,
                has shape (bs, num_dense_queries, 4) with the last
                dimension arranged as (x1, y1, x2, y2).
            gt_bboxes (list[Tensor]): List of unnormalized ground truth
                bboxes for each image, each has shape (num_gt, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
                NOTE: num_gt is dynamic for each image.
            gt_labels (list[Tensor]): List of ground truth classification
                index for each image, each has shape (num_gt,).
                NOTE: num_gt is dynamic for each image.
            img_metas (list[dict]): Meta information for one image,
                e.g., image size, scaling factor, etc.

        Returns:
            dict: A dictionary of loss components.
        """
        flatten_cls_scores = cls_scores
        flatten_bbox_preds = bbox_preds

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bbox_preds,
            gt_bboxes,
            img_metas,
            gt_labels_list=gt_labels,
        )
        (labels_list, label_weights_list, bbox_targets_list,
         alignment_metrics_list) = cls_reg_targets

        losses_cls, losses_bbox, \
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_single,
                flatten_cls_scores,
                flatten_bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                alignment_metrics_list,
                )

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(aux_loss_cls=losses_cls, aux_loss_bbox=losses_bbox)

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    img_metas,
                    gt_labels_list=None,
                    **kwargs):
        """Compute regression and classification targets for a batch images.

        Args:
            cls_scores (Tensor): Predicted normalized classification
                scores, has shape (bs, num_dense_queries,
                cls_out_channels).
            bbox_preds (Tensor): Predicted unnormalized bbox coordinates,
                has shape (bs, num_dense_queries, 4) with the last
                dimension arranged as (x1, y1, x2, y2).
            gt_bboxes_list (List[Tensor]): List of unnormalized ground truth
                bboxes for each image, each has shape (num_gt, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
                NOTE: num_gt is dynamic for each image.
            img_metas (list[dict]): Meta information for one image,
                e.g., image size, scaling factor, etc.
            gt_labels_list (list[Tensor]): List of ground truth classification
                    index for each image, each has shape (num_gt,).
                    NOTE: num_gt is dynamic for each image.
                    Default: None.

        Returns:
            tuple: a tuple containing the following targets.

            - all_labels (list[Tensor]): Labels for all images.
            - all_label_weights (list[Tensor]): Label weights for all images.
            - all_bbox_targets (list[Tensor]): Bbox targets for all images.
            - all_assign_metrics (list[Tensor]): Normalized alignment metrics
                for all images.
        """
        (all_labels, all_label_weights, all_bbox_targets,
         all_assign_metrics) = multi_apply(self._get_target_single, cls_scores,
                                           bbox_preds, gt_bboxes_list,
                                           gt_labels_list, img_metas)

        return (all_labels, all_label_weights, all_bbox_targets,
                all_assign_metrics)

    def _get_target_single(self, cls_scores, bbox_preds, gt_bboxes, gt_labels,
                           img_meta, **kwargs):
        """Compute regression and classification targets for one image.

        Args:
            cls_scores (Tensor): Predicted normalized classification
                scores for one image, has shape (num_dense_queries,
                cls_out_channels).
            bbox_preds (Tensor): Predicted unnormalized bbox coordinates
                for one image, has shape (num_dense_queries, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
            gt_bboxes (Tensor): Unnormalized ground truth
                bboxes for one image, has shape (num_gt, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
                NOTE: num_gt is dynamic for each image.
            gt_labels (Tensor): Ground truth classification
                    index for the image, has shape (num_gt,).
                    NOTE: num_gt is dynamic for each image.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels for one image.
            - label_weights (Tensor): Label weights for one image.
            - bbox_targets (Tensor): Bbox targets for one image.
            - norm_alignment_metrics (Tensor): Normalized alignment
                metrics for one image.
        """
        if len(gt_labels) == 0:
            num_valid_anchors = len(cls_scores)
            bbox_targets = torch.zeros_like(bbox_preds)
            labels = bbox_preds.new_full((num_valid_anchors, ),
                                         cls_scores.size(-1),
                                         dtype=torch.long)
            label_weights = bbox_preds.new_zeros(
                num_valid_anchors, dtype=torch.float)
            norm_alignment_metrics = bbox_preds.new_zeros(
                num_valid_anchors, dtype=torch.float)
            return (labels, label_weights, bbox_targets,
                    norm_alignment_metrics)

        assign_result = self.assigner.assign(cls_scores, bbox_preds, gt_bboxes,
                                             gt_labels, img_meta)
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics

        pred_instances = BaseDataElement()
        gt_instances = BaseDataElement()

        pred_instances.bboxes = bbox_preds
        gt_instances.bboxes = gt_bboxes

        pred_instances.priors = cls_scores
        gt_instances.labels = gt_labels

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = len(cls_scores)
        bbox_targets = torch.zeros_like(bbox_preds)
        labels = bbox_preds.new_full((num_valid_anchors, ),
                                     cls_scores.size(-1),
                                     dtype=torch.long)
        label_weights = bbox_preds.new_zeros(
            num_valid_anchors, dtype=torch.float)
        norm_alignment_metrics = bbox_preds.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            if gt_labels is None:
                # Only dense_heads gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]

            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = sampling_result.pos_assigned_gt_inds == gt_inds
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[
                pos_inds[gt_class_inds]] = pos_norm_alignment_metrics

        return (labels, label_weights, bbox_targets, norm_alignment_metrics)


class DDQAuxVPDLoss(nn.Module):
    """DDQ auxiliary branches loss for dense queries.

    Args:
        loss_cls (dict):
            Configuration of classification loss function.
        loss_bbox (dict):
            Configuration of bbox regression loss function.
        train_cfg (dict):
            Configuration of gt targets assigner for each predicted bbox.
    """

    def __init__(
        self,
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_dist=dict(type='JD', project=(-10, 10, 21), scale_alpha=1.0, skew_beta=0.2),
        train_cfg=dict(
            assigner=dict(type='TopkHungarianAssigner', topk=8),
            alpha=1,
            beta=6),
    ):
        super(DDQAuxVPDLoss, self).__init__()
        self.train_cfg = train_cfg
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_dist_cfg = loss_dist
        self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])

        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = TASK_UTILS.build(sampler_cfg)

    def loss_single(self, cls_score, bbox_pred, bbox_dist, bbox_samp, labels, label_weights,
                    bbox_targets, alignment_metrics, train_with_vpd):
        """Calculate auxiliary branches loss for dense queries for one image.

        Args:
            cls_score (Tensor): Predicted normalized classification
                scores for one image, has shape (num_dense_queries,
                cls_out_channels).
            bbox_pred (Tensor): Predicted unnormalized bbox coordinates
                for one image, has shape (num_dense_queries, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
            labels (Tensor): Labels for one image.
            label_weights (Tensor): Label weights for one image.
            bbox_targets (Tensor): Bbox targets for one image.
            alignment_metrics (Tensor): Normalized alignment metrics for one
                image.

        Returns:
            tuple: A tuple of loss components and loss weights.
        """
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, alignment_metrics)
        cls_loss_func = self.loss_cls

        loss_cls = cls_loss_func(
            cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = cls_score.size(-1)
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_decode_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_weight = alignment_metrics[pos_inds]

            if train_with_vpd:
                pos_decode_bbox_dist = bbox_dist[pos_inds]
                pos_decode_bbox_pred = bbox_samp[pos_inds]
                loss_bbox = self.loss_bbox(
                    pos_decode_bbox_pred,
                    pos_decode_bbox_targets,
                    weight=pos_bbox_weight,
                    avg_factor=1.0)
                # regularization loss
                loss_dist = self.loss_dist(
                    pos_decode_bbox_dist,
                    pos_decode_bbox_targets,
                    weight=pos_bbox_weight,
                    avg_factor=1.0)
                loss_bbox = loss_bbox + loss_dist
            else:
                pos_decode_bbox_pred = bbox_pred[pos_inds]
                loss_bbox = self.loss_bbox(
                    pos_decode_bbox_pred,
                    pos_decode_bbox_targets,
                    weight=pos_bbox_weight,
                    avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, alignment_metrics.sum(
        ), pos_bbox_weight.sum()

    def loss_dist(self, pred, target, weight, avg_factor):

        bbox_mean = pred[:, 0:4]
        bbox_lstd = pred[:, 4:8]
        bbox_prior = pred[:, 8:10]
        bbox_stride = pred[:, 10:11]
        bbox_target = target

        x1 = bbox_prior[..., 0] - bbox_target[..., 0]
        y1 = bbox_prior[..., 1] - bbox_target[..., 1]
        x2 = bbox_target[..., 2] - bbox_prior[..., 0]
        y2 = bbox_target[..., 3] - bbox_prior[..., 1]

        bbox_target = torch.stack([x1, y1, x2, y2], -1)

        bbox_target = torch.log((bbox_target / bbox_stride).clamp(min=EPS))

        loss = weight * self.regularization_loss(
            bbox_mean, bbox_lstd, bbox_target,
            self.loss_dist_cfg['type'], self.loss_dist_cfg['project'],
            self.loss_dist_cfg['scale_alpha'], self.loss_dist_cfg['skew_beta'], avg_factor)
        
        eps = torch.finfo(torch.float32).eps
        loss = loss.sum() / (avg_factor + eps)

        return loss

    def regularization_loss(self, mean, lstd, target, metric, project, scale_alpha, skew_beta, avg_factor):
        project = np.linspace(project[0], project[1], project[2])
        scale = (project.shape[0] - 1) / (project[-1] - project[0])
        acc = 1 / scale / 2
        target = (target.reshape(-1) - project[0]) * scale
        target = target.clamp(min=EPS, max=(project[-1]-project[0]) * scale-EPS)
        idx_left = target.long()
        idx_right = idx_left + 1
        weight_left = idx_right.float() - target
        weight_right = target - idx_left.float()
        # target distribution
        target_dist = weight_left.new_full((weight_left.shape[0], \
            project.shape[0]), 0, dtype=torch.float32)
        target_dist[torch.arange(target_dist.shape[0]), idx_left] = weight_left
        target_dist[torch.arange(target_dist.shape[0]), idx_right] = weight_right
        # predict distribution
        mean, lstd= mean.reshape(-1, 1), lstd.reshape(-1, 1)
        Qg = torch.distributions.normal.Normal(mean, lstd.exp())
        project = torch.tensor(project).type_as(mean).repeat(mean.shape[0],1)
        pred_dist = Qg.cdf(project + acc) - Qg.cdf(project - acc)
        # distribution distance
        if metric == 'JD':
            total_dist = pred_dist * (1 - skew_beta) + target_dist * skew_beta
            kl_p = pred_dist * torch.log((pred_dist + 1e-6) / (total_dist + 1e-6))
            total_dist = pred_dist * skew_beta  + target_dist * (1 - skew_beta)
            kl_q = target_dist * torch.log((target_dist + 1e-6) / (total_dist + 1e-6))
            loss = (kl_p + kl_q) / 2
        return loss.sum(1).reshape(-1, 4).sum(1) * scale_alpha
    
    def loss(self, cls_scores, bbox_preds, bbox_dists, bbox_samps, gt_bboxes, gt_labels, img_metas,
             **kwargs):
        """Calculate auxiliary branches loss for dense queries.

        Args:
            cls_scores (Tensor): Predicted normalized classification
                scores, has shape (bs, num_dense_queries,
                cls_out_channels).
            bbox_preds (Tensor): Predicted unnormalized bbox coordinates,
                has shape (bs, num_dense_queries, 4) with the last
                dimension arranged as (x1, y1, x2, y2).
            gt_bboxes (list[Tensor]): List of unnormalized ground truth
                bboxes for each image, each has shape (num_gt, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
                NOTE: num_gt is dynamic for each image.
            gt_labels (list[Tensor]): List of ground truth classification
                index for each image, each has shape (num_gt,).
                NOTE: num_gt is dynamic for each image.
            img_metas (list[dict]): Meta information for one image,
                e.g., image size, scaling factor, etc.

        Returns:
            dict: A dictionary of loss components.
        """
        train_with_vpd = kwargs.pop('train_with_vpd')
        assign_with_vpd = kwargs.pop('assign_with_vpd')

        flatten_cls_scores = cls_scores
        flatten_bbox_preds = bbox_preds
        flatten_bbox_dists = bbox_dists
        flatten_bbox_samps = bbox_samps

        if assign_with_vpd:
            cls_reg_targets = self.get_targets(
                flatten_cls_scores,
                flatten_bbox_samps,
                gt_bboxes,
                img_metas,
                gt_labels_list=gt_labels,
        )
        else:
            cls_reg_targets = self.get_targets(
                flatten_cls_scores,
                flatten_bbox_preds,
                gt_bboxes,
                img_metas,
                gt_labels_list=gt_labels,
            )
        (labels_list, label_weights_list, bbox_targets_list,
         alignment_metrics_list) = cls_reg_targets

        losses_cls, losses_bbox, \
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_single,
                flatten_cls_scores,
                flatten_bbox_preds,
                flatten_bbox_dists,
                flatten_bbox_samps,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                alignment_metrics_list,
                train_with_vpd,
                )

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(aux_loss_cls=losses_cls, aux_loss_bbox=losses_bbox)

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    img_metas,
                    gt_labels_list=None,
                    **kwargs):
        """Compute regression and classification targets for a batch images.

        Args:
            cls_scores (Tensor): Predicted normalized classification
                scores, has shape (bs, num_dense_queries,
                cls_out_channels).
            bbox_preds (Tensor): Predicted unnormalized bbox coordinates,
                has shape (bs, num_dense_queries, 4) with the last
                dimension arranged as (x1, y1, x2, y2).
            gt_bboxes_list (List[Tensor]): List of unnormalized ground truth
                bboxes for each image, each has shape (num_gt, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
                NOTE: num_gt is dynamic for each image.
            img_metas (list[dict]): Meta information for one image,
                e.g., image size, scaling factor, etc.
            gt_labels_list (list[Tensor]): List of ground truth classification
                    index for each image, each has shape (num_gt,).
                    NOTE: num_gt is dynamic for each image.
                    Default: None.

        Returns:
            tuple: a tuple containing the following targets.

            - all_labels (list[Tensor]): Labels for all images.
            - all_label_weights (list[Tensor]): Label weights for all images.
            - all_bbox_targets (list[Tensor]): Bbox targets for all images.
            - all_assign_metrics (list[Tensor]): Normalized alignment metrics
                for all images.
        """
        (all_labels, all_label_weights, all_bbox_targets,
         all_assign_metrics) = multi_apply(self._get_target_single, cls_scores,
                                           bbox_preds, gt_bboxes_list,
                                           gt_labels_list, img_metas)

        return (all_labels, all_label_weights, all_bbox_targets,
                all_assign_metrics)

    def _get_target_single(self, cls_scores, bbox_preds, gt_bboxes, gt_labels,
                           img_meta, **kwargs):
        """Compute regression and classification targets for one image.

        Args:
            cls_scores (Tensor): Predicted normalized classification
                scores for one image, has shape (num_dense_queries,
                cls_out_channels).
            bbox_preds (Tensor): Predicted unnormalized bbox coordinates
                for one image, has shape (num_dense_queries, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
            gt_bboxes (Tensor): Unnormalized ground truth
                bboxes for one image, has shape (num_gt, 4) with the
                last dimension arranged as (x1, y1, x2, y2).
                NOTE: num_gt is dynamic for each image.
            gt_labels (Tensor): Ground truth classification
                    index for the image, has shape (num_gt,).
                    NOTE: num_gt is dynamic for each image.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels for one image.
            - label_weights (Tensor): Label weights for one image.
            - bbox_targets (Tensor): Bbox targets for one image.
            - norm_alignment_metrics (Tensor): Normalized alignment
                metrics for one image.
        """
        if len(gt_labels) == 0:
            num_valid_anchors = len(cls_scores)
            bbox_targets = torch.zeros_like(bbox_preds)
            labels = bbox_preds.new_full((num_valid_anchors, ),
                                         cls_scores.size(-1),
                                         dtype=torch.long)
            label_weights = bbox_preds.new_zeros(
                num_valid_anchors, dtype=torch.float)
            norm_alignment_metrics = bbox_preds.new_zeros(
                num_valid_anchors, dtype=torch.float)
            return (labels, label_weights, bbox_targets,
                    norm_alignment_metrics)

        assign_result = self.assigner.assign(cls_scores, bbox_preds, gt_bboxes,
                                             gt_labels, img_meta)
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics

        pred_instances = BaseDataElement()
        gt_instances = BaseDataElement()

        pred_instances.bboxes = bbox_preds
        gt_instances.bboxes = gt_bboxes

        pred_instances.priors = cls_scores
        gt_instances.labels = gt_labels

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = len(cls_scores)
        bbox_targets = torch.zeros_like(bbox_preds)
        labels = bbox_preds.new_full((num_valid_anchors, ),
                                     cls_scores.size(-1),
                                     dtype=torch.long)
        label_weights = bbox_preds.new_zeros(
            num_valid_anchors, dtype=torch.float)
        norm_alignment_metrics = bbox_preds.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            if gt_labels is None:
                # Only dense_heads gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]

            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = sampling_result.pos_assigned_gt_inds == gt_inds
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[
                pos_inds[gt_class_inds]] = pos_norm_alignment_metrics

        return (labels, label_weights, bbox_targets, norm_alignment_metrics)