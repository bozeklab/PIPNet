import torch.nn as nn
import torch


class CKA_loss(nn.Module):
    def __init__(self, concept_cha):
        super(CKA_loss, self).__init__()
        self.concept_cha = concept_cha

    def __repr__(self):
        basic = super().__repr__()
        str_show = f"{basic[:-1]}concept_cha={self.concept_cha})"
        return str_show

    def unbiased_HSIC(self, x, y):
        # create the unit **vector** filled with ones
        n = x.shape[1]
        ones = torch.ones(x.shape[0], n, 1).cuda()

        # fill the diagonal entries with zeros
        # mask = torch.eye(n).repeat(x.shape[0], 1, 1).bool().cuda()
        mask = torch.eye(n).unsqueeze(0).cuda()
        x = x * (1 - mask)
        y = y * (1 - mask)

        # first part in the square brackets
        trace = torch.sum(torch.matmul(x, y.permute(0, 2, 1)) * mask, dim=(-1, -2), keepdim=True)

        # middle part in the square brackets
        nominator1 = torch.sum(x, dim=(-2, -1), keepdim=True)
        nominator2 = torch.sum(y, dim=(-2, -1), keepdim=True)
        denominator = (n - 1) * (n - 2)
        middle = torch.matmul(nominator1, nominator2) / denominator

        # third part in the square brackets
        multiplier1 = 2 / (n - 2)
        multiplier2 = torch.matmul(torch.matmul(ones.permute(0, 2, 1), x), torch.matmul(y, ones))
        last = multiplier1 * multiplier2

        # complete equation
        unbiased_hsic = 1 / (n * (n - 3)) * (trace + middle - last)
        return unbiased_hsic

    def CKA(self, kernel):
        index = torch.triu_indices(kernel.shape[0], kernel.shape[0], 1)
        nominator = self.unbiased_HSIC(kernel[index[0]], kernel[index[1]])
        denominator1 = self.unbiased_HSIC(kernel[index[0]], kernel[index[0]])
        denominator2 = self.unbiased_HSIC(kernel[index[1]], kernel[index[1]])
        denominator1 = torch.nn.functional.relu(denominator1)
        denominator2 = torch.nn.functional.relu(denominator2)
        denominator = denominator1 * denominator2
        # prevent divide 0
        # mask = (denominator != 0)
        cka = (nominator) / torch.sqrt(torch.clamp(denominator, min=1e-16))
        return cka

    def forward(self, feature_map):
        # calculate the concept number and channel number of each concept

        CKA_loss = 0
        #concept_num = concept_blocks.shape[1] // self.concept_cha[layer_i]
        cha_per_con = 1#self.concept_cha[layer_i]
        B, C, H, W = feature_map.shape
        sorted = []
        for p in range(C):
            pf = feature_map[:, p, :, :]
            #pf, _ = torch.sort(pf.view(B, H*W), dim=1)
            pf = pf.view(B, 1, H, W)
            sorted.append(pf)
        sorted = torch.cat(sorted, dim=1)
        concept_blocks = torch.flatten(sorted.reshape(B, C, cha_per_con, H, W).permute(1, 0, 2, 3, 4), 2)
        concept_blocks_kernel = torch.matmul(concept_blocks, concept_blocks.permute(0, 2, 1))
        CKA_loss = CKA_loss + torch.mean(torch.abs(self.CKA(concept_blocks_kernel)))
        return CKA_loss


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define concept channel sizes for different layers
    concept_cha = [4, 8]  # Example channel sizes per concept

    # Create an instance of the CKA_loss class
    cka_loss_fn = CKA_loss(concept_cha)

    # Simulate feature maps for two layers
    B, C1, H, W = 2, 32, 7, 7  # Batch size, channels, height, width for layer 1

    # Generate random feature maps for two layers
    layer1_features = torch.randn(B, C1, H, W)

    # Create a list of concept pools (one per layer)
    concept_pools = layer1_features

    # Compute the CKA loss
    loss = cka_loss_fn(concept_pools)

    print("CKA Loss:", loss.item())


if __name__ == "__main__":
    main()