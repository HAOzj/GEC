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
    field = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                       eos_token=EOS_WORD, pad_token=BLANK_WORD)

    # splits返回 splits of datasets objects
    # fields为data.Field类型
    fields = {"src": ("src", field), "trg": ("trg", field)}
    train = data.TabularDataset(path=OUTPUT_TRAIN_FILE, format='json', fields=fields,
                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                len(vars(x)['trg']) <= MAX_LEN)
    val = data.TabularDataset(path=OUTPUT_DEV_FILE, format='json', fields=fields,
                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                len(vars(x)['trg']) <= MAX_LEN)
    field.build_vocab(train, vectors="glove.6B.200d")
    vocab = field.vocab


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


if True:
    pad_idx = field.vocab.stoi["<blank>"]
    model = make_model(len(field.vocab), len(field.vocab), N=N)

    # 载入预训练的词向量
    model.src_embed[0].lut.weight.data.copy_(vocab.vectors)
    model.tgt_embed[0].lut.weight.data.copy_(vocab.vectors)

    criterion = LabelSmoothing(size=len(field.vocab), padding_idx=pad_idx, smoothing=0.1)

    train_iter = MyIterator(train, batch_size=BATCH_SIZE, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)


model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


for epoch in range(10):
    if True:
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model,
                  SimpleLossCompute(
                    model.generator, criterion,
                    opt=model_opt
                  )
        )

        torch.save(model.state_dict(), MODEL_PATH)
    else:
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


for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != field.vocab.stoi["<blank>"]).unsqueeze(-2)

    # 原句子
    print("Source:", end="\t")
    for i in range(1, src.size(1)):
        sys = field.vocab.itos[src[0, i]]
        if sys == "</s>": break
        print(sys, end=" ")

    # 预测
    out = greedy_decode(
        model, src, src_mask,
        max_len=MAX_LEN, start_symbol=field.vocab.stoi["<s>"])
    print("\n\nTranslation:", end="\t")
    for i in range(1, out.size(1)):
        sym = field.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end=" ")
    print()

    # 黄金答案
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = field.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end=" ")
    print()
    break

