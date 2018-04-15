import torch.nn as nn

from quicknlp.utils import get_list, concat_bidir_state, assert_dims, HParam


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
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
            nn.init.xavier_uniform(p)
    return model

class Transformer(nn.Module):
    """Transformer model based on https://arxiv.org/abs/1706.03762
        code implementation heavily inspired by http://nlp.seas.harvard.edu/2018/04/03/attention.html

    """

    def __init__(self, ntoken: HParam, emb_sz: HParam, nhid: HParam, nlayers: HParam, pad_token: int,
                 eos_token: int, max_tokens: int = 50, share_embedding_layer: bool = False, tie_decoder: bool = True,
                 bidir: bool = False, **kwargs):
        """

        Args:
            ntoken (Union[List[int],int]): Number of tokens for the encoder and the decoder
            emb_sz (Union[List[int],int]): Embedding size for the encoder and decoder embeddings
            nhid (Union[List[int],int]): Number of hidden dims for the encoder and the decoder
            nlayers (Union[List[int],int]): Number of layers for the encoder and the decoder
            pad_token (int): The  index of the token used for padding
            eos_token (int): The index of the token used for eos
            max_tokens (int): The maximum number of steps the decoder iterates before stopping
            share_embedding_layer (bool): if True the decoder shares its input and output embeddings
            tie_decoder (bool): if True the encoder and the decoder share their embeddings
            bidir (bool): if True use a bidirectional encoder
            **kwargs: Extra embeddings that will be passed to the encoder and the decoder
        """
        super(self).__init__()
        # allow for the same or different parameters between encoder and decoder
        ntoken, emb_sz, nhid, nlayers = get_list(ntoken, 2), get_list(emb_sz, 2), get_list(nhid, 2), get_list(nlayers,
                                                                                                              2)
        if "dropoutd" in kwargs:
            dropoutd = kwargs.pop("dropoutd")
        else:
            dropoutd = 0.5
        self.encoder = EmbeddingRNNEncoder(ntoken=ntoken[0], emb_sz=emb_sz[0], nhid=nhid[0], nlayers=nlayers[0],
                                           pad_token=pad_token, bidir=bidir, **kwargs)

        self.decoder = EmbeddingRNNDecoder(ntoken=ntoken[-1], emb_sz=emb_sz[-1], nhid=nhid[-1], nlayers=nlayers[-1],
                                           pad_token=pad_token, eos_token=eos_token, max_tokens=max_tokens,
                                           # Share the embedding layer between encoder and decoder
                                           embedding_layer=self.encoder.encoder_with_dropout.embed if share_embedding_layer else None,
                                           # potentially tie the output projection with the decoder embedding
                                           **kwargs
                                           )
        enc = self.decoder.encoder if tie_decoder else None
        self.decoder.projection_layer = Projection(n_out=ntoken[-1], n_in=emb_sz[-1], dropout=dropoutd,
                                                   tie_encoder=enc if tie_decoder else None
                                                   )
        self.nt = ntoken[-1]

    def forward(self, *inputs, num_beams=0):
        encoder_inputs, decoder_inputs = assert_dims(inputs, [2, None, None])  # dims: [sl, bs] for encoder and decoder
        # reset the states for the new batch
        bs = encoder_inputs.size(1)
        self.encoder.reset(bs)
        self.decoder.reset(bs)
        raw_outpus, outputs = self.encoder(encoder_inputs)
        state = concat_bidir_state(self.encoder.hidden)
        raw_outputs_dec, outputs_dec = self.decoder(decoder_inputs, hidden=state, num_beams=num_beams)
        if num_beams == 0:
            # use output of the projection module
            predictions = assert_dims(outputs_dec[-1], [None, bs, self.nt])  # dims: [sl, bs, nt]
        else:
            # use argmax or beam search predictions
            predictions = assert_dims(self.decoder.beam_outputs, [None, bs, num_beams])  # dims: [sl, bs, nb]
        return predictions, [*raw_outpus, *raw_outputs_dec], [*outputs, *outputs_dec]
