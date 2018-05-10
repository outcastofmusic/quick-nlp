***********
Quick NLP
***********


**Quick NLP**  is a deep learning nlp library inspired by the `fast.ai library  <https://github.com/fastai/fastai>`_

It follows the same api as fastai and extends it allowing for quick and easy running of nlp models

Features
--------

- Python 3.6 code
- Tight-knit integration with Fast.ai library:
    - Fast.ai style DataLoader objects for sentence to sentence algorithms
    - Fast.ai style DataLoader objects for dialogue algorithms
    - Fast.ai style DataModel objects for training nlp models
- Can run a seq2seq model with a few lines of code similar to existing fast.ai examples
- Easy to expand/train and try different models or use different data
- Ready made algorithms to try out
    - Seq2Seq https://arxiv.org/abs/1506.05869
    - Seq2Seq with Attention https://arxiv.org/abs/1703.03906
    - HRED http://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14567/14219
    - Attention is all you need http://papers.nips.cc/paper/7181-attention-is-all-you-need
    - Depthwise Separable Convolutions for Neural Machine Translation (TODO) https://arxiv.org/abs/1706.03059


Installation
------------

Installation of fast.ai library is required. Please install using the instructions `here <https://github.com/fastai/fastai>`_ .
It is important that the latest version of fast.ai is used and not the pip version which is not up to date.


After setting up an environment using the fasta.ai instructions please clone the quick-nlp repo and use pip install to install the package as follows:

.. code-block:: bash

    git clone https://github.com/outcastofmusic/quick-nlp
    cd quick-nlp
    pip install .


Docker Image
~~~~~~~~~~~~

A docker image with the latest master is available to use it please run:

.. code-block:: bash

    docker run --runtime nvidia -it -p 8888:8888 --mount type=bind,source="$(pwd)",target=/workspace agispof/quicknlp:latest

this will mount your current directory to /workspace and start a jupyter lab session in that directory

Usage Example
-------------

The main goal of quick-nlp is to provided the easy interface of the fast.ai library for seq2seq models.

For example  Lets assume that we have a dataset_path with folders for training, validation files.
Each file is a tsv file where each row is two sentences separated by a tab. For example a file inside the train folder can be a eng_to_fr.tsv file with the following first few lines::

    Go.	Va !
    Run!	Cours !
    Run!	Courez !
    Wow!	Ça alors !
    Fire!	Au feu !
    Help!	À l'aide !
    Jump.	Saute.
    Stop!	Ça suffit !
    Stop!	Stop !
    Stop!	Arrête-toi !
    Wait!	Attends !
    Wait!	Attendez !
    I see.	Je comprends.


loading the data from the directory is as simple as:

.. code-block:: python

    from fastai.plots import *
    from torchtext.data import Field
    from fastai.core import SGD_Momentum
    from fastai.lm_rnn import seq2seq_reg
    from quicknlp import SpacyTokenizer, print_batch, S2SModelData
    INIT_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    DATAPATH = "dataset_path"
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
                                        bs= batch_size
                                       )


Finally, to train a seq2seq model with the data we only need to do:

.. code-block:: python

    emb_size = 300
    nh = 1024
    nl = 3
    learner = data.get_model(opt_fn=SGD_Momentum(0.7), emb_sz=emb_size,
                             nhid=nh,
                             nlayers=nl,
                             bidir=True,
                            )
    clip = 0.3
    learner.reg_fn = reg_fn
    learner.clip = clip
    learner.fit(2.0, wds=1e-6)

