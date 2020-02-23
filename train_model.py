import torch
from torchtext import data, datasets
from component import (
    SimpleLossCompute, make_model, LabelSmoothing,
    run_epoch, NoamOpt, Batch, batch_size_fn,
    greedy_decode, iterative_decoding, beam_search
)
from generate_json_for_dataset import OUTPUT_FILE
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
    SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 500

    # splits返回 splits of datasets objects
    # fields为data.Field类型
    fields = {"src": ("src", SRC), "trg": ("trg", TGT)}
    ds = data.TabularDataset(path=OUTPUT_FILE, format='json', fields=fields)
    train, val = ds.split(split_ratio=[0.8, 0.2])
    MIN_FREQ = 2
    SRC.build_vocab(ds, min_freq=MIN_FREQ)
    TGT.build_vocab(ds, min_freq=MIN_FREQ)

    print("val的样例有", val.__len__())


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
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    BATCH_SIZE = 100
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    # valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
    #                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    #                         batch_size_fn=batch_size_fn, train=False)
    valid_iter = data.Iterator(val, batch_size=10, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            train=False)
    print("valid有:{}批".format(valid_iter.__len__()))


model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
PATH = "model/model_20200223"

for epoch in range(10):
    if False:
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model,
                  SimpleLossCompute(
                    model.generator, criterion,
                    opt=model_opt
                  )
        )

        torch.save(model.state_dict(), PATH)
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
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)

    # 原句子
    print("Source:", end="\t")
    for i in range(1, src.size(1)):
        sys = SRC.vocab.itos[src[0, i]]
        if sys == "</s>": break
        print(sys, end=" ")

    # 预测
    out = greedy_decode(
        model, src, src_mask,
        max_len=MAX_LEN, start_symbol=TGT.vocab.stoi["<s>"])
    print("\n\nTranslation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end=" ")
    print()

    # 黄金答案
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end=" ")
    print()
    break

