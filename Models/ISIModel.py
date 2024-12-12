import torch
import torch.nn as nn
import torch.nn.functional as F


class ISIModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, top_k):
        """
        Individual Cognition View-based Selected Interaction (ISI) 模块。
        :param input_dim: 输入维度（词向量维度）
        :param hidden_dim: 隐藏层维度
        :param top_k: 筛选出前k个差异最大的文章
        """
        super(ISIModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k

        # 文章句子级编码
        self.article_encoder = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)

        # 差异性权重矩阵
        self.Wm = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        self.Wn = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)

    def encode_articles(self, articles):
        """
        编码所有文章为句子级表示。
        :param articles: 文章序列 (batch_size, num_articles, article_len, input_dim)
        :return: 句子级表示 (batch_size, num_articles, 2*hidden_dim)
        """
        batch_size, num_articles, article_len, input_dim = articles.size()
        articles = articles.view(-1, article_len, input_dim)  # 合并batch与文章维度
        _, (hidden, _) = self.article_encoder(articles)  # hidden: (2, batch_size*num_articles, hidden_dim)
        hidden = torch.cat((hidden[0], hidden[1]), dim=-1)  # 拼接前向和后向的隐藏状态 (batch_size*num_articles, 2*hidden_dim)
        sentence_level_representations = hidden.view(batch_size, num_articles, -1)  # 恢复batch维度
        return sentence_level_representations

    def calculate_differences(self, sentence_reps):
        """
        计算文章间的差异性矩阵。
        :param sentence_reps: 句子级表示 (batch_size, num_articles, 2*hidden_dim)
        :return: 差异性矩阵 (batch_size, num_articles, num_articles)
        """
        um = torch.tanh(self.Wm(sentence_reps))
        un = torch.tanh(self.Wn(sentence_reps))

        # 计算差异性矩阵 A[m,n] = exp(u_m * u_n) / sum(exp(u_i * u_n))
        differences = torch.bmm(um, un.transpose(1, 2))  # (batch_size, num_articles, num_articles)
        attention_weights = F.softmax(differences, dim=-1)  # 按列归一化
        return attention_weights

    def select_top_k(self, sentence_reps, attention_weights):
        """
        筛选差异性最大的前 k 篇文章。
        :param sentence_reps: 句子级表示 (batch_size, num_articles, 2*hidden_dim)
        :param attention_weights: 差异性矩阵 (batch_size, num_articles, num_articles)
        :return: 筛选后的文章表示 (batch_size, top_k, 2*hidden_dim)
        """
        top_k_indices = torch.topk(-attention_weights.sum(dim=-1), self.top_k, dim=-1).indices  # 找出前k篇差异最大的文章
        batch_size = sentence_reps.size(0)
        selected_reps = torch.stack([
            sentence_reps[b, top_k_indices[b]] for b in range(batch_size)
        ])  # 按batch收集前k篇文章的表示
        return selected_reps

    def co_interaction(self, claim_encoded, selected_articles):
        # claim_encoded: (batch_size, 1, 2*hidden_dim)
        # selected_articles: (batch_size, top_k, 2*hidden_dim)

        claim_attention = F.softmax(torch.bmm(selected_articles, claim_encoded.transpose(1, 2)), dim=-1)
        # claim_attention: (batch_size, top_k, 1)
        # 将这个attention应用到claim上
        # claim只有1步，article_context就是selected_articles加权后的结果
        article_context = torch.bmm(claim_attention, claim_encoded)  # (batch_size, top_k, 2*hidden_dim)

        article_attention = F.softmax(torch.bmm(claim_encoded, selected_articles.transpose(1, 2)), dim=-1)
        # article_attention: (batch_size, 1, top_k)
        claim_context = torch.bmm(article_attention, selected_articles)  # (batch_size, 1, 2*hidden_dim)
        claim_context = claim_context.expand(-1, selected_articles.size(1), -1)  # (batch_size, top_k, 2*hidden_dim)

        local_evidence = torch.cat([article_context, claim_context], dim=-1)  # (batch_size, top_k, 4*hidden_dim)
        return local_evidence

    def forward(self, claim, articles):
        """
        前向传播。
        :param claim: 主张序列 (batch_size, claim_len, input_dim)
        :param articles: 文章序列 (batch_size, num_articles, article_len, input_dim)
        :return: 局部关键证据片段 (batch_size, top_k, 4*hidden_dim)
        """
        # 编码文章为句子级表示
        sentence_reps = self.encode_articles(articles)

        # 计算差异性并筛选前 k 篇文章
        differences = self.calculate_differences(sentence_reps)
        selected_articles = self.select_top_k(sentence_reps, differences)

        # 主张与筛选文章交互
        claim_encoded = self.encode_articles(claim.unsqueeze(1))  # 编码主张
        local_evidence = self.co_interaction(claim_encoded, selected_articles)

        return local_evidence
