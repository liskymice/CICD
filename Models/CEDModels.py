import torch
import torch.nn as nn
import torch.nn.functional as F


class ClaimGuidedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ClaimGuidedEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.article_encoder = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.claim_encoder = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)

        self.W1 = nn.Parameter(torch.Tensor(2 * hidden_dim, 2 * hidden_dim))
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.W1)

    def sequence_encoding(self, inputs, lstm):
        """
        对输入序列进行双向LSTM编码，返回整个序列的输出以及最终拼接隐藏状态
        inputs: (batch_size, seq_len, input_dim)
        output: (batch_size, seq_len, 2*hidden_dim)
        final_rep: (batch_size, 2*hidden_dim)
        """
        output, (h_n, _) = lstm(inputs)
        final_rep = torch.cat((h_n[0], h_n[1]), dim=-1)
        return output, final_rep

    def attention_matching(self, article_reps, claim_rep):
        """
        article_reps: (batch_size, num_articles, 2*hidden_dim)
        claim_rep: (batch_size, 2*hidden_dim)

        对多篇文章的句子级表示与claim进行attention聚合
        """
        # 扩展claim维度
        claim_rep_expanded = claim_rep.unsqueeze(-1) # (batch_size, 2*hidden_dim, 1)
        # 计算scores
        # article_reps @ W1: (batch_size, num_articles, 2*hidden_dim) * (2*hidden_dim, 2*hidden_dim) -> (batch_size, num_articles, 2*hidden_dim)
        tmp = torch.matmul(article_reps, self.W1)
        # 与claim_rep点积 (batch_size, num_articles, 2*hidden_dim) * (batch_size, 2*hidden_dim, 1)
        scores = torch.matmul(tmp, claim_rep_expanded).squeeze(-1) # (batch_size, num_articles)

        # 归一化
        attention_weights = F.softmax(scores, dim=1)
        # 聚合
        aggregated_rep = torch.bmm(attention_weights.unsqueeze(1), article_reps).squeeze(1) # (batch_size, 2*hidden_dim)
        return aggregated_rep

    def forward(self, claim, relevant_articles):
        """
        claim: (batch_size, claim_len, input_dim)
        relevant_articles: (batch_size, num_articles, article_len, input_dim)
        我们将对每篇文章编码得到 (batch_size, num_articles, 2*hidden_dim)
        并对多篇文章进行attention聚合

        返回：
        word_level_rep: 实际上可以返回每篇文章的整个输出序列，如需后续使用可以返回
        sentence_level_rep: 聚合后的句子级表示 (batch_size, 2*hidden_dim)
        """
        batch_size, num_articles, article_len, input_dim = relevant_articles.size()

        # 编码claim
        _, claim_rep = self.sequence_encoding(claim, self.claim_encoder)

        # 编码所有文章
        relevant_articles = relevant_articles.view(batch_size * num_articles, article_len, input_dim)
        article_outputs, article_final = self.sequence_encoding(relevant_articles, self.article_encoder)
        # 重塑回(batch_size, num_articles, ...)
        article_final = article_final.view(batch_size, num_articles, 2*self.hidden_dim)

        # 对多篇文章进行attention聚合
        aggregated_rep = self.attention_matching(article_final, claim_rep)

        # 返回文章序列输出和聚合表示
        # 如果需要词级表示（全文），可以同时返回article_outputs reshape后结果，这里根据需要返回final
        return article_final, aggregated_rep



class HierarchicalAttentionDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super(HierarchicalAttentionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.W2 = nn.Linear(hidden_dim, 1, bias=False)
        self.W3 = nn.Linear(hidden_dim, 1, bias=False)
        self.W4 = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def compute_attention_weights(self, decoder_hidden, memory_bank):
        expanded_hidden = decoder_hidden.unsqueeze(1).expand(-1, memory_bank.size(1), -1)
        scores = torch.tanh(memory_bank + expanded_hidden)
        scores = self.W2(scores)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(scores.squeeze(-1), dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), memory_bank).squeeze(1)
        return context_vector, attention_weights

    def forward(self, word_level_rep, sentence_level_rep, max_len):
        batch_size = word_level_rep.size(0)
        device = word_level_rep.device

        decoder_hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        decoder_cell = torch.zeros(batch_size, self.hidden_dim, device=device)
        generated_sequence = torch.zeros(batch_size, max_len, self.vocab_size, device=device)

        for t in range(max_len):
            sentence_context, _ = self.compute_attention_weights(decoder_hidden, sentence_level_rep.unsqueeze(1))
            word_context, _ = self.compute_attention_weights(decoder_hidden, word_level_rep)

            combined_context = torch.cat([sentence_context, word_context], dim=-1)
            decoder_input = torch.tanh(self.W4(combined_context)).unsqueeze(1)  # (batch_size, 1, hidden_dim)

            _, (decoder_hidden_new, decoder_cell_new) = self.decoder_lstm(decoder_input, (decoder_hidden.unsqueeze(0), decoder_cell.unsqueeze(0)))
            decoder_hidden = decoder_hidden_new.squeeze(0)
            decoder_cell = decoder_cell_new.squeeze(0)

            decoder_output = self.output_layer(decoder_hidden)
            generated_sequence[:, t, :] = decoder_output

        # 将最后一步的decoder_hidden作为G_rep
        G_rep = decoder_hidden
        return generated_sequence, G_rep



class CED(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(CED, self).__init__()
        self.encoder = ClaimGuidedEncoder(input_dim, hidden_dim)
        # 对2*hidden_dim投射到hidden_dim
        self.proj_word = nn.Linear(2*hidden_dim, hidden_dim, bias=False)
        self.proj_sent = nn.Linear(2*hidden_dim, hidden_dim, bias=False)
        self.decoder = HierarchicalAttentionDecoder(hidden_dim, vocab_size)

    def forward(self, claim, relevant_articles, max_len):
        # 编码阶段
        word_level_rep, aggregated_rep = self.encoder(claim, relevant_articles)
        # word_level_rep: (batch_size, total_words, 2*hidden_dim)
        # aggregated_rep: (batch_size, 2*hidden_dim)

        word_level_rep = self.proj_word(word_level_rep) # (batch_size, total_words, hidden_dim)
        sentence_level_rep = self.proj_sent(aggregated_rep) # (batch_size, hidden_dim)

        # 解码阶段
        generated_sequence, G_rep = self.decoder(word_level_rep, sentence_level_rep, max_len)
        # generated_sequence: (batch_size, max_len, vocab_size)
        # G_rep: (batch_size, hidden_dim) 最后一步decoder hidden作为G的表示向量

        return generated_sequence, G_rep



