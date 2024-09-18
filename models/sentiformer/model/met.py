from hmac import new
# from torchinfo import summary

import torch
from torch import nn, sigmoid, softmax
from .met_layer import Transformer, CrossTransformer, HhyperLearningEncoder
# from .bert import BertTextEncoder
from einops import repeat


class MET(nn.Module):
    def __init__(self, dataset, ARL_depth=3, fusion_layer_depth=6, bert_pretrained='bert-base-uncased'):
        super(MET, self).__init__()

        self.h_hyper = nn.Parameter(torch.ones(1, 8, 128))

        # self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=bert_pretrained)

        self.proj_l0 = nn.Linear(512, 128)
        self.proj_a0 = nn.Linear(512, 128)
        self.proj_v0 = nn.Linear(512, 128)

        self.proj_l = Transformer(num_frames=1, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_a = Transformer(num_frames=1, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_v = Transformer(num_frames=1, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        # self.proj_l1 = nn.Sequential(
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048,8)
        # )
        # self.proj_a1 = nn.Sequential(
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048,8)
        # )
        # self.proj_v1 = nn.Sequential(
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048,8)
        # )
        # self.mlp_l = nn.Sequential(
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1024)
        # )
        # self.mlp_a = nn.Sequential(
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1024)
        # )
        # self.mlp_v = nn.Sequential(
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1024)
        # )


        self.text_encoder = Transformer(num_frames=8, save_hidden=True, token_len=None, dim=128, depth=ARL_depth-1, heads=8, mlp_dim=128)
        self.h_hyper_layer = HhyperLearningEncoder(dim=128, depth=ARL_depth, heads=8, dim_head=16, dropout = 0.)
        self.fusion_layer = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=128, depth=fusion_layer_depth, heads=8, mlp_dim=128)
        # self.mlp_fusion = nn.Sequential(
        #     nn.Linear(2048, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128)
        # )
        # self.heu_prompt_confidence = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1),
        #     nn.Sigmoid()
        # )
        # self.visual_confidence = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1),
        #     nn.Sigmoid()
        # )
        # self.text_confidence = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1),
        #     nn.Sigmoid()
        # )
        self.cls_head = nn.Sequential(
            nn.Linear(128, 8),
            nn.Softmax(dim=-1)
        )


    def forward(self, x_visual, x_heu_prompt, x_text):
        b = x_visual.size(0)
        x_heu_prompt = x_heu_prompt.unsqueeze(1)
        x_text = x_text.unsqueeze(1)
        x_visual = x_visual.unsqueeze(1)
        
        # h_v_cp = x_visual.clone().detach()
        # h_a_cp = x_heu_prompt.clone().detach()
        # h_t_cp = x_text.clone().detach()
        
        # x_text.fill_(0)

        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b = b)

        # x_text = self.bertmodel(x_text)

        x_visual = self.proj_v0(x_visual)
        x_heu_prompt = self.proj_a0(x_heu_prompt)
        x_text = self.proj_l0(x_text)

        h_v = self.proj_v(x_visual)
        h_a = self.proj_a(x_heu_prompt)
        h_t = self.proj_l(x_text)
        
        # h_v = self.proj_v1(h_v.reshape(b,-1))
        # h_a = self.proj_a1(h_a.reshape(b,-1))
        # h_t = self.proj_l1(h_t.reshape(b,-1))
        
        

        # h_v_confidence = self.visual_confidence(h_v_cp).squeeze(-1)
        # h_a_confidence = self.heu_prompt_confidence(h_a_cp).squeeze(-1)
        # h_t_confidence = self.text_confidence(h_t_cp).squeeze(-1)

        # hvc_holo = torch.log(h_v_confidence)/(torch.log(h_a_confidence*h_t_confidence*h_v_confidence)+1e-8)
        # hac_holo = torch.log(h_a_confidence)/(torch.log(h_a_confidence*h_t_confidence*h_v_confidence)+1e-8)
        # htc_holo = torch.log(h_t_confidence)/(torch.log(h_a_confidence*h_t_confidence*h_v_confidence)+1e-8)
        # cbt = h_t_confidence.detach()+htc_holo.detach()
        # cbv = h_v_confidence.detach()+hvc_holo.detach()
        # cba = h_a_confidence.detach()+hac_holo.detach()
        # w_all = torch.stack([cba,cbv,cbt], dim=-1)

        # w_all = nn.Softmax(dim=-1)(w_all).reshape(b,-1)

        # w_a = w_all[:,0]
        # w_v = w_all[:,1]
        # w_t = w_all[:,2]

        # print(w_a.shape)
        # exit(0)        
        # h_a = h_a*(w_a.detach())
        # h_v = h_v*(w_v.detach())
        # h_t = h_t*(w_t.detach())
        # h_a = torch.einsum('b,bi->bi', w_a.detach(), h_a)
        # h_v = torch.einsum('b,bi->bi', w_v.detach(), h_v)
        # h_t = torch.einsum('b,bi->bi', w_t.detach(), h_t)
        
        # h_v = self.mlp_v(x_visual.reshape(b,-1)).reshape(b,8,-1)
        # h_a = self.mlp_a(x_heu_prompt.reshape(b,-1)).reshape(b,8,-1)
        # h_t = self.mlp_l(x_text.reshape(b,-1)).reshape(b,8,-1)

        h_t_list = self.text_encoder(h_t)

        h_hyper = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper)     
        feat = self.fusion_layer(h_hyper, h_t_list[-1])[:, 0]
        # new_h_hyper = torch.concat([h_hyper.reshape(h_hyper.shape[0],-1), h_t_list[-1].reshape(h_hyper.shape[0],-1)], dim=-1)

        # feat = self.mlp_fusion(new_h_hyper)
        output = self.cls_head(feat)
        # output = h_a+h_v+h_t

        return output


def build_model(opt):
    model = MET(dataset = opt.datasetName, fusion_layer_depth=opt.fusion_layer_depth, bert_pretrained = 'bert-base-uncased')
    return model
