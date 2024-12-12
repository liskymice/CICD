from Models.CEDModels import *
from Models.ISIModel import ISIModule
from torch import nn
import torch

class CICD(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, top_k, num_classes, lambda_kl=0.1, max_len=10):
        super(CICD, self).__init__()
        self.ced = CED(input_dim, hidden_dim, vocab_size)
        self.isi = ISIModule(input_dim, hidden_dim, top_k)
        self.num_classes = num_classes
        self.lambda_kl = lambda_kl
        self.max_len = max_len

        # 分类器参数，Wg, Wi, Wp
        # G_rep: (batch_size, hidden_dim)
        # I: (batch_size, top_k, 4*hidden_dim) -> flatten: (batch_size, top_k*4*hidden_dim)
        # [G; I] -> (batch_size, hidden_dim + top_k*4*hidden_dim)

        # 假设top_k和hidden_dim已知
        self.top_k = top_k
        self.classifier_G = nn.Linear(hidden_dim, num_classes)
        self.classifier_I = nn.Linear(top_k*4*hidden_dim, num_classes)
        self.classifier_all = nn.Linear(hidden_dim + top_k*4*hidden_dim, num_classes)

    def forward(self, claim, relevant_articles, labels=None):
        # CED得到全局证据G
        generated_sequence, G_rep = self.ced(claim, relevant_articles, self.max_len)
        # ISI得到局部证据I
        local_evidence = self.isi(claim, relevant_articles)
        # local_evidence: (batch_size, top_k, 4*hidden_dim)

        batch_size = local_evidence.size(0)
        I_rep = local_evidence.view(batch_size, self.top_k*4*self.ced.encoder.hidden_dim)

        # 基于G的分类分布 p_G
        p_G = F.softmax(self.classifier_G(G_rep), dim=-1) # (batch_size, num_classes)
        # 基于I的分类分布 p_I
        p_I = F.softmax(self.classifier_I(I_rep), dim=-1) # (batch_size, num_classes)
        # 基于[G;I]的分类分布 p_all
        GI_rep = torch.cat([G_rep, I_rep], dim=-1)
        p_all = F.softmax(self.classifier_all(GI_rep), dim=-1) # (batch_size, num_classes)

        if labels is not None:
            # 计算损失
            # 1. 分类损失 (使用p_all和真实标签的交叉熵)
            loss_sup = F.nll_loss(torch.log(p_all), labels)

            # 2. 不一致性损失 KL(p_G || p_I)
            # KL散度: sum p_G * log(p_G/p_I)
            # p_G和p_I已经是softmax输出
            kl_loss = F.kl_div(p_I.log(), p_G, reduction='batchmean')
            # 需要确认KL方向性，这里用 p_I.log()和p_G对调则为Dkl(p_G||p_I)
            # 原论文定义：Loss_in = D_kl(G||I) = sum G_i log(G_i/I_i)
            # 所以应使用p_G进行log，p_I为参考分布：
            kl_loss = F.kl_div(p_I.log(), p_G, reduction='batchmean')
            # 由于Pytorch的KL散度函数定义是 KLDivLoss(input||target)=sum target * (log(target)-input)
            # 我们需要 Dkl(G||I)=sum G * log(G/I) = sum G * (logG - logI)
            # 对于F.kl_div(input, target): input=log_probs, target=probs
            # 我们想要Dkl(G||I) = sum G log(G/I) = sum G (logG - logI)
            # 等价: Dkl(G||I) = sum G logG - sum G logI
            # 使用F.kl_div(x.log(), y)不直接给出这个式子
            # 这里简化，假设 Dkl(G||I) = F.kl_div(logI, G) 或检查文献
            # 实际上：F.kl_div输入log_probs, 第二参数是目标分布
            # 若想要Dkl(G||I)=sum G log(G/I), 按PyTorch文档:
            # D_kl(P||Q)= sum P(x)*[log P(x)-log Q(x)]
            # 我们传入: input=log P_G, target=P_I
            kl_loss = F.kl_div(p_G.log(), p_I, reduction='batchmean') # 确保是Dkl(G||I)

            # 总损失
            loss = loss_sup + self.lambda_kl * kl_loss

            return p_all, loss
        else:
            return p_all, None