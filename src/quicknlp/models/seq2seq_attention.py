from quicknlp.modules import AttentionDecoder, AttentionProjection, Encoder, RNNEncoder
from quicknlp.modules.embeddings import DropoutEmbeddings
from quicknlp.utils import HParam, assert_dims, get_kwarg
from .seq2seq import Seq2Seq, get_list


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
        ntoken, emb_sz, nhid, nlayers = get_list(ntoken, 2), get_list(emb_sz, 2), \
                                        get_list(nhid, 2), get_list(nlayers, 2)
        dropoutd = get_kwarg(kwargs, name="dropoutd", default_value=0.5)
        dropoute = get_kwarg(kwargs, name="dropout_e", default_value=0.1)
        dropouti = get_kwarg(kwargs, name="dropout_i", default_value=0.65)
        dropouth = get_kwarg(kwargs, name="dropout_h", default_value=0.3)
        wdrop = get_kwarg(kwargs, name="wdrop", default_value=0.5)
        cell_type = get_kwarg(kwargs, name="cell_type", default_value="lstm")
        encoder_embedding_layer = DropoutEmbeddings(ntokens=ntoken[0],
                                                    emb_size=emb_sz[0],
                                                    dropoute=dropoute,
                                                    dropouti=dropouti
                                                    )

        encoder_rnn = RNNEncoder(in_dim=emb_sz[0],
                                 out_dim=kwargs.get("out_dim", emb_sz[0]),
                                 nhid=nhid[0], bidir=bidir,
                                 dropouth=dropouth,
                                 wdrop=wdrop,
                                 nlayers=nlayers[0],
                                 cell_type=cell_type,
                                 )
        self.encoder = Encoder(
            embedding_layer=encoder_embedding_layer,
            encoder_layer=encoder_rnn
        )

        if share_embedding_layer:
            decoder_embedding_layer = encoder_embedding_layer
        else:
            decoder_embedding_layer = DropoutEmbeddings(ntokens=ntoken[-1],
                                                        emb_size=emb_sz[-1],
                                                        dropoute=dropoute,
                                                        dropouti=dropouti
                                                        )

        decoder_rnn = RNNEncoder(in_dim=kwargs.get("in_dim", emb_sz[-1] * 2), out_dim=kwargs.get("out_dim", emb_sz[-1]),
                                 nhid=nhid[-1], bidir=False, dropouth=dropouth,
                                 wdrop=wdrop, nlayers=nlayers[-1], cell_type=cell_type)

        projection_layer = AttentionProjection(out_dim=ntoken[-1],
                                               in_dim=emb_sz[-1],
                                               dropout=dropoutd,
                                               att_nhid=att_nhid,
                                               tie_encoder=decoder_embedding_layer if tie_decoder else None
                                               )
        self.nlayers = nlayers
        self.decoder = AttentionDecoder(
            decoder_layer=decoder_rnn,
            projection_layer=projection_layer,
            embedding_layer=decoder_embedding_layer,
            pad_token=pad_token,
            eos_token=eos_token,
            max_tokens=max_tokens,
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
