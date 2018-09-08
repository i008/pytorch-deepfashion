import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AdaptiveMaxPool2d
from torchvision.models import vgg16

ROI_POOL_SIZE = (3, 3)
N_LANDMARKS = 8


def adaptive_max_pool(input, size):
    return AdaptiveMaxPool2d(size[0], size[1])(input)


def gated_roi_pooling(input, rois, gates=None, size=ROI_POOL_SIZE, spatial_scale=1.0):
    """
    Standard roi-pooling extended to accept a mask vector (gates) wich will set all activations to zero for
    correspnding features

    :param input: features (for instance  feature maps from vgg/resnet
    :param rois: rois [batch_id, x1, y1, x2, y2]
    :param gates: mask vector with shape [len(gates), 1]
    :param size: size of the pooled regions (for instance (3,3)
    :param spatial_scale:
    :return:
    """
    assert (rois.dim() == 2)
    assert (rois.size(1) == 5)
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)

    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
        mp = adaptive_max_pool(im, size)[0]
        output.append(mp)

    pooled_features = torch.cat(output, 0)
    if gates is not None:
        pooled_features[gates.flatten()] = 0

    return pooled_features.view(input.shape[0], -1, ROI_POOL_SIZE[0], ROI_POOL_SIZE[1])


def landmark_predictions_to_roipool_boxes(landmark_loc, bbs=0):
    """

    Basically what we are doing here is translating the landmark location predictions (basically a vector of [BS, 16])
    Into a format digestable by the roi_pooling functions. We require to rescale the predictions to the correpsonging
    feature map size (in case of this VGG network we rescale by 16)

    After we have the landmarks we need to create "boxes" around them this is controlled by the bbs parameter

    :param landmark_loc:  [BS, 16]
    :param bbs: size of the bounding box
    :return:
    """
    collect = []
    for batch_ix, landmark_position in enumerate(landmark_loc.reshape(-1, 2).unsqueeze(0)):
        for landmark_pair in landmark_position:
            x, y = landmark_pair
            x, y = int(x / 16), int(y / 16)
            box = torch.Tensor([batch_ix, x - bbs, y - bbs, x + bbs, y + bbs])
            box.clamp_(min=0)
            collect.append(box)
    to_roi_pool = torch.stack(collect)
    return to_roi_pool


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class FashionNetVgg16NoBn(nn.Module):
    def __init__(self):
        super(FashionNetVgg16NoBn, self).__init__()
        vgg = vgg16()

        features = list(vgg.features.children())
        self.conv4 = nn.Sequential(*features[:-8])  # the paper implements DF with features taken from conv4 from vgg16
        self.conv5_pose = nn.Sequential(*features[-8:])
        self.conv5_global = nn.Sequential(*features[-8:])

        self.fc6_global = nn.Linear(in_features=512 * 7 * 7, out_features=1024 * 4)
        self.fc6_local = nn.Linear(in_features=512 * N_LANDMARKS * ROI_POOL_SIZE[0] * ROI_POOL_SIZE[1],
                                   out_features=1024)
        self.fc6_pose = nn.Linear(in_features=512 * 7 * 7, out_features=1024)
        self.fc7_pose = nn.Linear(in_features=1024, out_features=1024)
        self.loc = nn.Linear(in_features=1024, out_features=16)
        self.vis = nn.Linear(in_features=1024, out_features=8)

        #  5120 = concat[(4*1024)+1024]
        self.massive_attr = nn.Linear(in_features=5120, out_features=1000)
        self.categories = nn.Linear(in_features=5120, out_features=50)

        self.flatten = Flatten()

    def forward(self, x):
        base_features = self.conv4(x)
        pose = self.flatten(self.conv5_pose(base_features))

        pose = F.leaky_relu(self.fc6_pose(pose))
        pose = F.leaky_relu(self.fc7_pose(pose))

        pose_loc = self.loc(pose)
        pose_vis = F.sigmoid(self.vis(pose))

        # bbs is about bow big the roi box is going to be (area around landmark)
        roi_boxes = landmark_predictions_to_roipool_boxes(pose_loc, bbs=3)
        # here you have to decide about the gating policy
        pools = gated_roi_pooling(base_features, roi_boxes, pose_vis < -1000)

        fc6_local = F.leaky_relu(self.fc6_local(self.flatten(pools)))
        fc6_global = F.leaky_relu(self.fc6_global(self.flatten(self.conv5_global(base_features))))

        global_and_local = torch.cat([fc6_local, fc6_global], dim=1)

        massive_attr = F.sigmoid(self.massive_attr(global_and_local))
        categories = F.sigmoid(self.categories(global_and_local))

        return massive_attr, categories


if __name__ == '__main__':
    fn = FashionNetVgg16NoBn()

    # pose network needs to be trained from scratch? i guess?
    for k in fn.state_dict().keys():
        if 'conv5_pose' in k and 'weight' in k:
            torch.nn.init.xavier_normal_(fn.state_dict()[k])
            print('filling xavier {}'.format(k))

    for k in fn.state_dict().keys():
        if 'conv5_global' in k and 'weight' in k:
            torch.nn.init.xavier_normal_(fn.state_dict()[k])
            print('filling xavier {}'.format(k))

    img = torch.Tensor(torch.rand(2, 3, 224, 224))

    fn(img)
