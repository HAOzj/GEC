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
    MODEL_PATH, CORPORA
)

import warnings
warnings.filterwarnings("ignore")

GENE_FLAG = False  # 是否把损失写入文件
DRAW_FLAG = False  # 是否把文件中的损失plot
WRITE_FLAG = True  #
TRAIN_FLAG = True
TRAIN_FILE = f"train_loss_{CORPORA}.txt"
VAL_FILE = f"val_loss_{CORPORA}.txt"

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

    
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                    torch.optim.Adam(model.parameters(),
                                     lr=0, betas=(0.9, 0.98),
                                     eps=1e-9)
                    )


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


def plot_loss_curve(end, write_flag, train_file, valid_file):
    """计算loss curve

    Args:
        end(int) :- plot的模型对应的最后的epoches数
        write_flag(bool) :- 训练集和发展集上的损失是否写入文件
        train_file(path) :- 训练集上的损失写入的文件
        valid_file(path) :- 发展集上的损失写入的文件
    """
    epoches = []
    loss_train, loss_valid = [], []
    for epoch in range(end):
        print(f"开始{epoch+1}轮")
        model.load_state_dict(
            torch.load(f"{MODEL_PATH}_{epoch}_bucket"),
            strict=False
        )
        model.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in train_iter),
                         model,
                         LossCompute(model.generator, criterion)
        )
        epoches.append(epoch+1)
        loss_train.append(loss)
        if write_flag:
            with open(train_file, "a+") as fp:
                fp.write(f"{epoch+1}_{loss}\n")

        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model,
                         SimpleLossCompute(model.generator, criterion)
        )
        loss_valid.append(loss)
        if write_flag:
            with open(valid_file, "a+") as fp:
                fp.write(f"{epoch+1}_{loss}\n")


def plot_file(train_file, val_file):
    epoches, loss_train, loss_valid = [], [], []
    with open(train_file, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            ele = line.split("_")
            epoches.append(int(ele[0]))
            loss_train.append(float(ele[1]))
    with open(val_file, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            ele = line.split("_")
            loss_valid.append(float(ele[1]))
    import matplotlib.pyplot as plt

    plt.plot(epoches, loss_train, label="loss on training set")
    plt.plot(epoches, loss_valid, label="loss on valid set")
    for epoch, loss1, loss2 in zip(epoches, loss_train, loss_valid):
        if epoch % 10 == 0:
            plt.annotate(f'({round(loss1, 3)})', xy=(epoch, loss1),
                         xytext=(epoch, (loss1+loss2)/2),
                         arrowprops=dict(facecolor='black', shrink=0.05))
            plt.annotate(f'({round(loss2, 3)})', xy=(epoch, loss2), xytext=(epoch, loss2+0.2),
                         arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlabel("epoch")
    plt.ylabel("loss per token")
    plt.xlim((0, max(epoches)))
    plt.ylim((0, 2))
    plt.legend()
    plt.show()


if GENE_FLAG:
    plot_loss_curve(60, WRITE_FLAG, TRAIN_FILE, VAL_FILE)

if DRAW_FLAG:
    plot_file(TRAIN_FILE, VAL_FILE)


if TRAIN_FLAG:
    for epoch in range(60):
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
        with open(TRAIN_FILE, "a+") as fp:
            fp.write(f"{epoch}_{loss}\n")

        torch.save(model.state_dict(), MODEL_PATH+"_{}_lang8".format(epoch))
        model.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model,
                         SimpleLossCompute(
                             model.generator, criterion,
                             opt=None
                         )
        )
        with open(VAL_FILE, "a+") as fp:
            fp.write(f"{epoch}_{loss}\n")
        model.train()

    import sys
    sys.exit()
else:
    PATH = f"{MODEL_PATH}_59_bucket"
    model.load_state_dict(torch.load(PATH), strict=False)


ITERATIVE_FLAG = False
if ITERATIVE_FLAG:
    print("使用iterative decoding")

i = 0
for batch in valid_iter:
    src = batch.src.transpose(0, 1)[i:i+1]
    tgt = batch.trg.transpose(0, 1)[i:i+1]
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


