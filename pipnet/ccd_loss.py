
import torch.nn as nn
import torch


class CCD_loss(nn.Module):
    def __init__(self,  margin=0.01, eps=1e-8):
        super(CCD_loss, self).__init__()
        self.margin = margin
        self.relu = torch.nn.ReLU()
        self.eps = eps

    def __repr__(self):
        basic = super().__repr__()
        #str_show = f"{basic[:-1]}concept_cha={self.concept_cha}, margin={self.margin})"
        return basic

    def KL_div(self, x, y):
        print('KL ', x.shape, y.shape, x)
        #print(x, torch.log2(x), torch.log2(y))
        return torch.sum(x * (torch.log2(x) - torch.log2(y)), dim=-1)

    def JS_div(self, img_node, cd_node, eps=1e-8):
        img_node = img_node + eps
        cd_node = cd_node + eps
        img_node = img_node.unsqueeze(1)
        cd_node = cd_node.unsqueeze(0)
        M = (img_node + cd_node) / 2
        #print('KL val: ', self.KL_div(img_node, M))
        return (self.KL_div(img_node, M) + self.KL_div(cd_node, M)) / 2

    def forward(self, feat,  label, class_MCP_dist):
        #max_responses = []
        concept_num = feat.shape[1]
        cha_per_con = 1
        B, C, H, W = feat.shape
        feat = feat.reshape(B, concept_num, cha_per_con, H, W)
        #feat = feat - concept_mean[layer[layer_i] - 1].unsqueeze(0).unsqueeze(3).unsqueeze(4)
        #feat_norm = feat / (torch.norm(feat, dim=2, keepdim=True) + self.eps)

        # calculate concept vector from covariance matrix
        #response = torch.sum(feat_norm * concept_vector[layer[layer_i] - 1].unsqueeze(0).unsqueeze(3).unsqueeze(4),
        #                     dim=2)
        #max_response, max_index = torch.nn.functional.adaptive_max_pool2d(response, output_size=1,
        #                                                                  return_indices=True)
        #max_responses.append(torch.clip((max_response[..., 0, 0] + 1) / 2, min=self.eps, max=1))

        argmax_indices = torch.argmax(feat, dim=1, keepdim=True)
        mask = torch.zeros_like(feat).scatter_(1, argmax_indices, 1)
        proto_features = feat * mask
        print('!! proto_features.shape')
        pf_s, _ = torch.sort(proto_features, dim=1)
        proto_features = pf_s.squeeze(dim=2).flatten(2)

        #img_MCP_dist = torch.cat(max_responses, dim=1)
        #img_MCP_dist = img_MCP_dist / torch.sum(img_MCP_dist, dim=-1, keepdim=True)
        #print('!!! ', proto_features.shape, class_MCP_dist.shape)
        MCP_dist = self.JS_div(proto_features, class_MCP_dist)
        #print('!!! ', proto_features.shape, class_MCP_dist.shape)
        print('!!! MCP_dist ', MCP_dist.shape)
        print('!!! ', torch.gather(MCP_dist, dim=1, index=label[:, None, None]).shape)
        #print('!!! label ', label[:, None, None].shape)
        same_class = torch.mean(torch.gather(MCP_dist, dim=1, index=label[:, None, None]))
        mask = torch.ones_like(MCP_dist)
        mask[(torch.arange(label.shape[0]), label)] = 0
        diff_dist = self.relu((self.margin - MCP_dist) * mask)
        denominator = torch.sum(diff_dist != 0, dim=1)
        # prevent divided by zero
        denominator[denominator == 0] = 1
        diff_class = torch.mean(torch.sum(diff_dist, dim=1) / denominator)
        total_loss = (same_class + diff_class)
        print(total_loss)
        return total_loss


def main():
    pass


if __name__ == "__main__":
    main()
