_base_ = './crowddet_1svpd_JD_p2_40_a1.0_b0.2.py'

model = dict(roi_head=dict(bbox_head=dict(with_refine=True)))
