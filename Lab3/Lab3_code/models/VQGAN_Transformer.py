import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import torch.nn.functional as F
#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        # 1. 讀取權重檔案 
        state_dict = torch.load(load_ckpt_path, map_location='cpu', weights_only=False)
        
        # 2. 準備一個新的空字典
        new_state_dict = {}
        
        # 3. 反向翻譯：把 query_linear 變回 W_q
        for key, value in state_dict.items():
            new_key = key
            # 注意這裡！前面是檔案裡的名字(query)，後面是模型要的名字(W_q)
            new_key = new_key.replace('.query_linear.', '.W_q.')
            new_key = new_key.replace('.key_linear.', '.W_k.')
            new_key = new_key.replace('.value_linear.', '.W_v.')
            new_key = new_key.replace('.proj_linear.', '.W_o.')
            
            new_state_dict[new_key] = value

        # 4. 把翻譯好的字典載入模型
        self.transformer.load_state_dict(new_state_dict, strict=True)
        print("✅ 成功透過【反向】名稱轉換載入權重！")
    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        z_mapping, z_indices,_=self.vqgan.encode(x)
        #[B,256]
        z_indices=z_indices.view(z_mapping.shape[0],-1)
        return z_mapping, z_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x,ratio):
        #[B,256]
        _, z_indices = self.encode_to_z(x)
       # mask_rate = self.gamma(ratio)
        device = z_indices.device
        B, N = z_indices.shape
        num_masked = math.ceil(ratio * N)
        rand_prob = torch.rand(B, N, device=device)
        mask_pos = torch.topk(rand_prob, num_masked, dim=1).indices
        masked_z_indices = z_indices.clone()
        masked_z_indices.scatter_(-1, mask_pos, self.mask_token_id)
        logits = self.transformer(masked_z_indices)
        logits = logits[..., :self.mask_token_id]
        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, masked_z_indices, ratio, mask, total_mask_num):
        z_indices_input = torch.where(mask, self.mask_token_id, masked_z_indices)
        logits = self.transformer(z_indices_input)

        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = logits[..., :self.mask_token_id]
        probs = F.softmax(logits, dim=-1)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(probs, dim=-1)

        #predicted probabilities add temperature annealing gumbel noise as confidence
        u = torch.rand_like(z_indices_predict_prob)
        g = -torch.log(-torch.log(u + 1e-9))  # gumbel noise
        ratio=self.gamma(ratio)
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        mask = mask.bool()
        confidence[~mask] = float('inf')

        #define how much the iteration remain predicted tokens by mask scheduling
        mask_rate = self.gamma(ratio)
        mask_num = math.ceil(mask_rate * total_mask_num)

        #sort the confidence for the rank 
        mask_bc = torch.zeros_like(mask)
        if mask_num > 0:
            mask_pos = torch.topk(confidence, mask_num, dim=-1, largest=False).indices
            mask_bc.scatter_(-1, mask_pos, True)

        ##At the end of the decoding process, add back the original(non-masked) token values
        z_indices_predict[~mask] = masked_z_indices[~mask]

        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
