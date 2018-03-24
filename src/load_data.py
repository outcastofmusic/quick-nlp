from torch.optim import Adam
from torchtext.data import Field

from fastai.core import SGD_Momentum, to_gpu
from fastai.model import fit
from quicknlp import SpacyTokenizer, S2SModelData
from quicknlp.modules import Seq2Seq

INIT_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
DATAPATH = "/home/agis/PycharmProjects/pytorch-seq2seq/dataset"
fields = [
    ("english", Field(init_token=INIT_TOKEN, eos_token=EOS_TOKEN, tokenize=SpacyTokenizer('en'), lower=True)),
    ("french", Field(init_token=INIT_TOKEN, eos_token=EOS_TOKEN, tokenize=SpacyTokenizer('fr'), lower=True))

]
batch_size = 64
data = S2SModelData.from_text_files(path=DATAPATH, fields=fields,
                                    train="train",
                                    validation="validation",
                                    source_names=["english", "french"],
                                    target_names=["french"],
                                    bs=batch_size
                                    )
print(f'num tr batches: {len(data.trn_dl)}, num tr samples: {len(data.trn_ds)}')
print(f'num val batches: {len(data.val_dl)},num val samples: {len(data.val_ds)}')

emb_size = 300
nh = 1024
nl = 3
learner = data.get_model(opt_fn=SGD_Momentum(0.7), emb_sz=emb_size,
                         nhid=nh,
                         nlayers=nl,
                         max_tokens=20,
                         projection=""
                         )
# reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
clip = 0.3
# learner.reg_fn = reg_fn
# learner.clip = clip
print("untrained model")
# print_batch(lr=learner, dt=data, input_field="english", output_field="french", num_sentences=4)

# learner.fit(7, 100, wds=1e-6, cycle_len=4)
# learner.save("overfitting_seq2seq")
# learner.load("overfitting_seq2seq")

print("trained model")
# print_batch(lr=learner, dt=data, input_field="english", output_field="french", num_sentences=4)
ntoken = [data.nt[name] for name in data.trn_dl.source_names]
model = Seq2Seq(ntoken=ntoken, nhid=nh, nlayers=1, emb_sz=emb_size, pad_token=data.pad_idx,
                eos_token=data.eos_idx)
model = to_gpu(model)
fit(model=model, data=data, epochs=1, opt=Adam(params=model.parameters()), crit=learner.s2sloss)
