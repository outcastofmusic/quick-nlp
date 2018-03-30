from quicknlp.utils import assert_dims
from .seq2seq import Seq2Seq, HParam, get_list
from .submodules import RNNAttentionDecoder, EmbeddingRNNEncoder, AttentionProjection


class Seq2SeqAttention(Seq2Seq):

    def __init__(self, ntoken: HParam, emb_sz: HParam, nhid: HParam, nlayers: HParam, att_nhid: int, pad_token: int,
                 eos_token: int, max_tokens: int = 50, share_embedding_layer: bool = False, tie_decoder: bool = True,
                 bidir: bool = False, **kwargs):
        """

        Args:
            ntoken (Union[List[int],int]): Number of tokens for the encoder and the decoder
            emb_sz (Union[List[int],int]): Embedding size for the encoder and decoder embeddings
            nhid (Union[List[int],int]): Number of hidden dims for the encoder and the decoder
            nlayers (Union[List[int],int]): Number of layers for the encoder and the decoder
            att_nhid (int): Number of hidden dims for the attention Module
            pad_token (int): The  index of the token used for padding
            eos_token (int): The index of the token used for eos
            max_tokens (int): The maximum number of steps the decoder iterates before stopping
            share_embedding_layer (bool): if True the decoder shares its input and output embeddings
            tie_decoder (bool): if True the encoder and the decoder share their embeddings
            bidir (bool): if True use a bidirectional encoder
            **kwargs: Extra embeddings that will be passed to the encoder and the decoder
        """
        super(Seq2Seq, self).__init__()
        # allow for the same or different parameters between encoder and decoder
        ntoken, emb_sz, nhid, nlayers = get_list(ntoken), get_list(emb_sz), get_list(nhid), get_list(nlayers)
        if "dropoutd" in kwargs:
            dropoutd = kwargs.pop("dropoutd")
        else:
            dropoutd = 0.5
        self.encoder = EmbeddingRNNEncoder(ntoken=ntoken[0], emb_sz=emb_sz[0], nhid=nhid[0], nlayers=nlayers[0],
                                           pad_token=pad_token, bidir=bidir, **kwargs)

        self.decoder = RNNAttentionDecoder(ntoken=ntoken[-1], emb_sz=emb_sz[-1], nhid=nhid[-1], nlayers=nlayers[-1],
                                           pad_token=pad_token, eos_token=eos_token, max_tokens=max_tokens,
                                           # Share the embedding layer between encoder and decoder
                                           embedding_layer=self.encoder.encoder_with_dropout.embed if share_embedding_layer else None,
                                           # potentially tie the output projection with the decoder embedding
                                           **kwargs
                                           )
        enc = self.decoder.encoder if tie_decoder else None
        self.decoder.projection_layer = AttentionProjection(n_out=ntoken[-1],
                                                            n_in=emb_sz[-1],
                                                            dropout=dropoutd,
                                                            att_nhid=att_nhid,
                                                            tie_encoder=enc if tie_decoder else None
                                                            )
        self.nlayers = nlayers
        self.nhid = nhid
        self.emb_sz = emb_sz

    def forward(self, *inputs, num_beams=0):
        encoder_inputs, decoder_inputs = inputs
        # reset the states for the new batch
        bs = encoder_inputs.size(1)
        self.encoder.reset(bs)
        self.decoder.reset(bs)
        raw_outpus, outputs = self.encoder(encoder_inputs)
        state = self.decoder.hidden
        assert_dims(outputs, [self.nlayers[0], None, bs, (self.nhid[0], self.emb_sz[0])])
        # pass the encoder outputs as keys to the attention projection_layer
        self.decoder.projection_layer.reset(keys=outputs[-1])
        raw_outputs_dec, outputs_dec = self.decoder(decoder_inputs, hidden=state, num_beams=num_beams)
        # outputs_dec[-1].shape ==  (sl, bs, num_tokens)
        predictions = outputs_dec[-1] if num_beams == 0 else self.decoder.beam_outputs
        return predictions, [*raw_outpus, *raw_outputs_dec], [*outputs, *outputs_dec]
