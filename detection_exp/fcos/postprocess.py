import torch
from .bounding_box import BoxList
from .boxlist_ops import cat_boxlist, boxlist_ml_nms
from .fcos_targets import BBOX_SCALE, BBOX_BINS, compute_locations
import sys
import logging
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


PRE_NMS_THRESH = 0.05
PRE_NMS_TOP_N =  1000
NMS_THRESH = 0.6
FPN_POST_NMS_TOP_N = 100
NHEAD = 5


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        num_classes: int,
        pre_nms_thresh = PRE_NMS_THRESH,
        pre_nms_top_n = PRE_NMS_TOP_N,
        nms_thresh = NMS_THRESH,
        fpn_post_nms_top_n = FPN_POST_NMS_TOP_N,
        bbox_aug_enabled=False,
        discrete_bbox_pred=False
    ):
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled
        self.discrete_bbox_pred = discrete_bbox_pred
        self.BBOX_BINS = torch.nn.Parameter(BBOX_BINS.clone())

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """

        N, C, H, W = box_cls.shape
        logger.debug(f"N, C, H, W: {N}, {C}, {H}, {W}")

        logger.debug(f"location.shape: {locations.shape}")

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C)  #.sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        # force centerness input to be non-neg? Shouldn't be needed due to threshold
        centerness = centerness.reshape(N, -1)  #.sigmoid()

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        logger.debug("Forward_for_single_feature_map:")
        logger.debug(f"box_regression: {box_regression}")
        logger.debug(f"box_cls: {box_cls}")
        logger.debug(f"N: {N}")
        logger.debug(f"candidate_inds.shape: {candidate_inds.shape}")
        
        results = []
        # iter over images
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            logger.debug(f"per_candidate_inds: {per_candidate_inds}")
            logger.debug(f"per_candidate_nonzeros: {per_candidate_nonzeros}") 
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]
            logger.debug(i)
            logger.debug(f"pre_nms_top_n: {pre_nms_top_n}")
            logger.debug(f"per_pre_nms_top_n: {per_pre_nms_top_n}")


            ndet_pre_nms = per_candidate_inds.sum().item()
            if ndet_pre_nms > per_pre_nms_top_n.item():
                logger.debug(f"There are {ndet_pre_nms} detections pre_nms. Trimming to top K.")
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            logger.debug(f"per_locations: {per_locations}")
            logger.debug(f"per_box_regression: {per_box_regression}")
            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist = boxlist.clip_to_image(remove_empty=False)
            results.append(boxlist)

        return results
        
    # take in dict of predicted marginals and call forward()
    def postprocess_ngmpred(self, predicted_marginals, graph, img_height, img_width, true_img_dimensions=None):
        logger.debug(f"predicted_marginals.keys(): {predicted_marginals.keys()}")
        nimg = predicted_marginals[graph["Bbox_0"]].mean.shape[0]

        box_regression = []
        for hd in range(NHEAD):
            if self.discrete_bbox_pred:
                #bbox = torch.movedim(predicted_marginals[graph[f"Bbox_{hd}"]].probs, 1, -1)
                bbox = predicted_marginals[graph[f"Bbox_{hd}"]].probs
                logger.debug(f"pp ngmpred bbox shape: {bbox.shape}")
                bbox = torch.sum(bbox*self.BBOX_BINS, dim=-1)
                bbox = bbox * BBOX_SCALE
                if torch.isnan(bbox).any():
                    raise ValueError(f"nan in bbox: {bbox}")
                box_regression.append(bbox)
            else:
                box_regression.append(predicted_marginals[graph[f"Bbox_{hd}"]].mean*BBOX_SCALE)
        
        # no longer 'unlabeled' class to chop off
        #box_cls = [predicted_marginals[graph[f"BboxCls_{hd}"]].probs[:,:,:,1:].movedim(-1,1) for hd in range(NHEAD)]
        box_cls = [predicted_marginals[graph[f"BboxCls_{hd}"]].probs for hd in range(NHEAD)]
        if self.discrete_bbox_pred:
            centerness = [predicted_marginals[graph[f"Centerness_{hd}"]].probs for hd in range(NHEAD)]
        else:
            centerness = [predicted_marginals[graph[f"Centerness_{hd}"]].mean for hd in range(NHEAD)]

        image_sizes = [(img_height, img_width) for i in range(nimg)]
        
        logger.debug(f"postprocess_ngmpred box_regression: {box_regression}")
        locations = compute_locations(img_height, img_width, device=box_regression[0].device)
        results = self.forward(locations, box_cls, box_regression, centerness, image_sizes)

        if true_img_dimensions:
            # resize output to size that the answers are meant to be at, from the size the NN saw
            results = [boxlist.resize(true_img_dimensions) for boxlist in results]

        return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes
                )
            )

        logger.debug(f"boxes out of forward_for_single_feature_map:")
        for boxlist in sampled_boxes:
            logger.debug(boxlist[0].bbox)
        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not self.bbox_aug_enabled:
            boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                logger.debug("Selecting down to top K after nms.")
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


