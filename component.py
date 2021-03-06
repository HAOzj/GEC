import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from conf_loader import (
    MAX_LEN, DROPOUT, BEAM_WIDTH,
    D_FF, D_MODEL, H, N
)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()

        # self-attn + ffn
        # 每一层都要先layerNorm,进入self-attn或ffn,然后dropout,输出再residual
        # N层后还要layerNorm
        self.encoder = encoder

        self.decoder = decoder # self-attn + encoder-decoder attn + ffn
        self.src_embed = src_embed  # positional embedding + word embedding
        self.tgt_embed = tgt_embed
        self.generator = generator # d_model -> V

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences.

        src_mask用在encoder-decoder attn
        tgt_mask用在decoder的self attn
        src_mask是mask掉blank, tgt在src_mask基础上再mask掉subsequent tokens
        """
        return self.decode(
            memory=self.encode(src, src_mask),
            src_mask=src_mask,
            tgt=tgt, tgt_mask=tgt_mask
        )

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """Layer normalization.

    input and output are of the same dim
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # features取值为d_model
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') # triu上三角矩阵,k=1使得主对角线也为0
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=DROPOUT):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        # contiguous保证了tensor底层一维数组储存顺序是行优先的,
        # transpose和view等操作会影响一致性,再contiguous则会重新开辟内存
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k) 
             
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    def __init__(self, d_model, dropout, max_len=MAX_LEN):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # dim=[max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # buffer不同于parameters不会被bp更新,但也会保存下来
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=N,
               d_model=D_MODEL, d_ff=D_FF, h=H, dropout=DROPOUT):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# Batches and Masking
class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


# Training Loop
def run_epoch(data_iter, model, loss_compute):
    """Standard Training and Logging Function

    model outputs a layer-normed d_model-dim vector
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 49:
            elapsed = time.time() - start
            print("Batch Num: %d Loss: %f Tokens per Sec: %f Num of tokens: %d" %
                    (i+1, loss / batch.ntokens, tokens / elapsed, batch.ntokens))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


# Training Data and Batching
global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding.

    Arguments: 
        new: new example to add.
        count: current count of examples in the batch.
        sofar: current effective batch size.
    """
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


# Optimizer
class NoamOpt:
    """Optim wrapper that implements rate."""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# Label Smoothing
class LabelSmoothing(nn.Module):
    """Implement label smoothing."""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()

        # KLDivLoss需要传入prob
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size # 取值为vocab_size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))

        # torch.Tensor.scatter_(dim: int, index: LongTensor, src: Tensor or float)表示分散
        # dim=0时, self[index[i][j][k]] [j] [k] = src[i][j][k],也就是index的i-th row表示src的i-th row每个元素分配到which row的相同位置
        # dim=1时, self[i] [index[i][j][k]] [k] = src[i][j][k],也就是index的j-th column表示src的j-th column每个元素分到which column的相同位置
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)  # torch.nonzero返回所有非0元素的坐标,这行是把tgt的blank mask掉
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


# Synthetic Data
def data_gen(V, batch, nbatches):
    """Generate random data for a src-tgt copy task."""
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1   # BOE为1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


# Loss Computation
class SimpleLossCompute:
    """A simple loss compute and train function."""
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        """计算mini batch的损失

        Args:
            x: d_model,经过generator变成V-dim
            y: 真实index
            norm: 在这里是ntokens

        Returns:
            该batch的总loss
        """
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


# Greedy Decoding
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1]) # 最后一个位置
        _, next_word = torch.max(prob, dim=1) # max返回(values, indices)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def beam_search(model, src, src_mask, max_len, start_symbol, beam_width=BEAM_WIDTH):
    memory = model.encode(src, src_mask)
    """Beam search.
    
    output the best <beam_width> answers with their probs
    """
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    beams = [(ys, 1)]
    for _ in range(max_len-1):
        beams_tmp = []
        for ys in beams:
            ys = ys[0]
            out = model.decode(
                memory, src_mask, Variable(ys),
                Variable(subsequent_mask(ys.size(1)).type_as(src.data))
            )
            prob = model.generator(out[:, -1])
            values, indices = torch.topk(prob, k=beam_width, dim=1)
            for i in range(beam_width):
                beams_tmp.append((
                        torch.cat([ys, 
                            torch.ones(1, 1).type_as(src.data).fill_(indices.data[0][i])], dim=1),
                        values[0][i].item()
                    )
                )
        beams = sorted(beams_tmp, key=lambda x: x[1], reverse=True)[: beam_width]
    return beams


def iterative_decoding(model, src, src_mask, max_len, start_symbol, beam_width=BEAM_WIDTH, threshold=1, padding_idx=0):
    memory = model.encode(src, src_mask)
    """Iterative decoding.
    
    Based off multi-round of beam search, select the best non-identity answer 
    until the prob of the best answer is higher than the cost of the identity times <threshold>
    """
    INF = 10000

    while True:
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        beams = [(ys, 1)]
    
        for _ in range(max_len-1):
            beams_tmp = []
            for ys in beams:
                ys = ys[0]
                out = model.decode(
                    memory, src_mask, Variable(ys),
                    Variable(subsequent_mask(ys.size(1)).type_as(src.data))
                )
                prob = model.generator(out[:, -1])
                values, indices = torch.topk(prob, k=beam_width, dim=1)
                for i in range(beam_width):
                    beams_tmp.append((
                            torch.cat([ys, 
                                torch.ones(1, 1).type_as(src.data).fill_(indices.data[0][i])], dim=1),
                            values[0][i].item()
                        )
                    )
            beams = sorted(beams_tmp, key=lambda x: x[1], reverse=True)[: beam_width]

        loss_identity = -INF
        loss_non = - INF
        H_non = None 
        for beam in beams:
            if beam[0].equal(src.data):
                loss_identity = beam[1]
            elif beam[1] > loss_non:
                loss_non = beam[1]
                H_non = beam[0]

        # 如果需要rewrites
        # 改变 src, src_mask
        if loss_non > threshold * loss_identity:
            src = H_non
            # src_mask = src != padding_idx
            # src_mask = Variable(src_mask)
        else:
            break 
        
    return src


def main():
    V = 11
    PATH = "model/model_artificial"
    if True:
        criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
        model = make_model(V, V, N=2)
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        for epoch in range(10):
            model.train()
            run_epoch(data_gen(V, 30, 20), model, 
                    SimpleLossCompute(model.generator, criterion, model_opt))
            model.eval()
            run_epoch(data_gen(V, 30, 5), model, 
                    SimpleLossCompute(model.generator, criterion, None))
        model.eval()


        # Save and Load
        torch.save(model.state_dict(), PATH)

    modelB = make_model(V, V, N=2)
    modelB.load_state_dict(torch.load(PATH), strict=False)


    # Test
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    print(iterative_decoding(modelB, src, src_mask, max_len=10, start_symbol=1))


if __name__ == "__main__":
    main()

