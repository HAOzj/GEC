import torch
from torchtext import data
from component import (
    SimpleLossCompute, make_model, LabelSmoothing,
    run_epoch, NoamOpt, Batch, batch_size_fn,
    greedy_decode, iterative_decoding, beam_search
)
from conf_loader import (
    MAX_LEN, BATCH_SIZE, N,
    OUTPUT_TRAIN_FILE, OUTPUT_DEV_FILE,
    MODEL_PATH
)

import warnings
warnings.filterwarnings("ignore")


if True:
    import spacy

    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]


    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"

    # data.Field返回一个文本处理的datatype
    SRC = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    # splits返回 splits of datasets objects
    # fields为data.Field类型
    fields = {"src": ("src", SRC), "trg": ("trg", SRC)}
    train = data.TabularDataset(path=OUTPUT_TRAIN_FILE, format='json', fields=fields,
                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                len(vars(x)['trg']) <= MAX_LEN)
    val = data.TabularDataset(path=OUTPUT_DEV_FILE, format='json', fields=fields,
                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                len(vars(x)['trg']) <= MAX_LEN)
    SRC.build_vocab(train, vectors="glove.6B.200d")
    vocab = SRC.vocab


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    """Fix order in torchtext to match ours"""
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


class LossCompute:
    """A loss compute and train function."""
    def __init__(self, generator, criterion, opt=None):
        """计算batch损失.

        因为generator加了softmax, 所以用了KLDivLoss,如果没有softmax就用CrossEntropyLoss
        Args:
            generator(nn.Linear) :- linear + softmax
        """
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        """计算一个batch的样例

        x应该第二个维度是 V

        Args:
            x: d_model-dim
            y: 真实的index
            norm: 在这里是ntokens

        Returns:
            该样例的总loss
        """
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.item() * norm


if True:
    pad_idx = SRC.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(SRC.vocab), N=N)
    print("vocab_size={}".format(len(SRC.vocab)))

    # 载入预训练的词向量
    model.src_embed[0].lut.weight.data.copy_(vocab.vectors)
    model.tgt_embed[0].lut.weight.data.copy_(vocab.vectors)

    criterion = LabelSmoothing(size=len(SRC.vocab), padding_idx=pad_idx, smoothing=0.1)

    train_iter = data.BucketIterator(train, batch_size=BATCH_SIZE, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = data.BucketIterator(val, batch_size=BATCH_SIZE, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            train=False)


def check_ntokens(iter):
    for batch in iter:
        print(batch.src.size())
        src = batch.src.transpose(0, 1)
        ntokens = 0
        for i in range(1, src.size(0)):
            print("\n{}个数据".format(i))
            for j in range(1, src.size(1)):
                sys = SRC.vocab.itos[src[i, j]]
                if sys == "</s>": continue
                ntokens += 1
                print(sys, end=" ")
        break
    print("一共有{}tokens".format(ntokens))


# check_ntokens(train_iter)
# import sys
# sys.exit()
    
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


if True:
    def plot(x, y1, y2):
        """plot loss curve.

        How loss goes with time.
        """
        import matplotlib.pyplot as plt
        plt.plot(x, y1, label="loss on training set")
        plt.plot(x, y2, label="loss on valid set")
        plt.xlabel("epoch")
        plt.ylabel("loss per token")
        plt.legend()
        plt.show()

    x = []
    loss_train = []
    loss_valid = []
    for epoch in range(10):
        model.train()
        print("第{}个epoch".format(epoch))
        loss = run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model,
                  LossCompute(
                      model.generator, criterion,
                      opt=torch.optim.Adam(
                          params=model.parameters()
                      )
                  )
        )
        x.append(epoch)
        loss_train.append(loss)

        torch.save(model.state_dict(), MODEL_PATH+"_{}_bucket".format(epoch))
        model.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model,
                         SimpleLossCompute(
                             model.generator, criterion,
                             opt=None
                         )
        )
        loss_valid.append(loss)
        model.train()


    plot(x, loss_train, loss_valid)
else:
    PATH = f"{MODEL_PATH}_9"
    model.load_state_dict(torch.load(PATH), strict=False)


model.eval()
loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                    model,
                    SimpleLossCompute(
                        model.generator, criterion,
                        opt=None
                     )
                 )
print("验证集上loss为: ", loss)
import sys
sys.exit()

ITERATIVE_FLAG = True
if ITERATIVE_FLAG:
    print("使用iterative decoding")

for batch in valid_iter:
    src = batch.src.transpose(0, 1)[1:2]
    tgt = batch.trg.transpose(0, 1)[1:2]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)

    # 原句子
    print("\nSource:", end="\t")
    for i in range(1, src.size(1)):
        sys = SRC.vocab.itos[src[0, i]]
        if sys == "</s>": break
        print(sys, end=" ")

    # 预测
    if ITERATIVE_FLAG:
        out = iterative_decoding(
            model, src, src_mask,
            max_len=MAX_LEN, start_symbol=SRC.vocab.stoi["<s>"]
        )
    else:
        out = beam_search(
            model, src, src_mask,
            max_len=MAX_LEN, start_symbol=SRC.vocab.stoi["<s>"])[0][0]
    print("\nCorrection:", end="\t")
    for i in range(1, out.size(1)):
        sym = SRC.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end=" ")
    print()

    # 黄金答案
    print("Target:", end="\t")
    for i in range(1, tgt.size(1)):
        sym = SRC.vocab.itos[tgt[0, i]]
        if sym == "</s>": break
        print(sym, end=" ")
    print()


