from fastai.imports import *
from fastai.plots import *
from torchtext.data import Field
from fastai.lm_rnn import seq2seq_reg
from quicknlp import print_batch, HierarchicalModelData
from pathlib import Path

path = Path.home() / "PycharmProjects/personal/quick-nlp/tutorials/dataset/dialogue"

ubuntu_data = pd.read_csv(path / "train/dialogueText.csv")

field = Field(tokenize="spacy", lower="True")
cols = {"text_col": "text", "batch_col": "dialogueID", "sort_col": "date", "role_col": "from"}
ubuntu_data['dialogueID'].nunique()
sampled_dialogues = set(ubuntu_data['dialogueID'].sample(1000).tolist())
train_df = ubuntu_data.loc[ubuntu_data['dialogueID'].apply(lambda x: x in sampled_dialogues)]
# train_df = ubuntu_data

model_data = HierarchicalModelData.from_dataframes(path=path,
                                                   train_df=train_df,
                                                   val_df=train_df,
                                                   text_field=field,
                                                   **cols
                                                   )

learner = model_data.get_model()

reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
clip = 0.3
learner.reg_fn = reg_fn
learner.clip = clip

learner.fit(lrs=0.01, n_cycle=2, cycle_len=2)

print_batch(lr=learner, dt=model_data, input_field="text", output_field="text", num_sentences=4)
