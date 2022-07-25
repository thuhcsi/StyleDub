from copy import deepcopy

_init = True

class Options(dict):

    def __getitem__(self, key):
        if _init and not key in self.keys():
            self.__setitem__(key, Options())
        return super().__getitem__(key)

    def __getattr__(self, attr):
        if _init and not attr in self.keys():
            self[attr] = Options()
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        del self[attr]

    def __deepcopy__(self, memo=None):
        new = Options()
        for key in self.keys():
            new[key] = deepcopy(self[key])
        return new

baseline = Options()
baseline.max_epochs = 50
baseline.batch_size = 32
baseline.learning_rate = 1e-4

baseline.input.bert_dim = 768
baseline.input.sbert_dim = 768
baseline.input.gst_dim = 40
baseline.input.lst_dim = 40

baseline.local_encoder.input_dim = baseline.input.bert_dim + baseline.input.lst_dim
baseline.local_encoder.prenet.sizes = [256, 128]
baseline.local_encoder.cbhg.dim = 128
baseline.local_encoder.cbhg.K = 16
baseline.local_encoder.cbhg.projections = [128, 128]
baseline.local_encoder.output_dim = baseline.local_encoder.cbhg.dim * 2

baseline.local_text_encoder = deepcopy(baseline.local_encoder)
baseline.local_text_encoder.input_dim = baseline.input.bert_dim

baseline.attention.dim = 128
baseline.attention.k1_dim = baseline.local_text_encoder.output_dim
baseline.attention.k2_dim = baseline.attention.k1_dim
baseline.attention.preserved_k1_dim = baseline.local_encoder.output_dim
baseline.attention.preserved_k2_dim = baseline.attention.preserved_k1_dim

baseline.gst_linear_1.input_dim = 2 * baseline.input.sbert_dim + baseline.input.gst_dim
baseline.gst_linear_1.output_dim = baseline.input.gst_dim
baseline.gst_linear_2 = deepcopy(baseline.gst_linear_1)

baseline.lst_linear_1.input_dim = baseline.local_encoder.output_dim + baseline.local_text_encoder.output_dim
baseline.lst_linear_1.output_dim = baseline.input.lst_dim
baseline.lst_linear_2 = deepcopy(baseline.lst_linear_1)

proposed = deepcopy(baseline)

_init = False
