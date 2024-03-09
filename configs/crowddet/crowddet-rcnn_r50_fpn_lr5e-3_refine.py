_base_ = './crowddet-rcnn_r50_fpn_lr5e-3.py'

model = dict(roi_head=dict(bbox_head=dict(with_refine=True)))
