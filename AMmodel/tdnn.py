import tensorflow as tf

from AMmodel.layers.time_frequency import Melspectrogram
from AMmodel.transducer_wrap import Transducer
from AMmodel.ctc_wrap import CtcModel
from AMmodel.las_wrap import LAS, LASConfig
from utils.tools import merge_two_last_dims


class TimeDelayLayer(tf.keras.layers.Layer):
    def __init__(self,
                 output_channels: int,
                 context: int,
                 dilate: int,
                 dropout: float = 0,
                 pad: str = "same",
                 name: str = "TDNN_layer",
                 **kwargs
                 ):
        super(TimeDelayLayer, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv1D(
            filters=output_channels, 
            kernel_size=context,
            dilation_rate=dilate,
            padding=pad
        )
        self.bn = tf.keras.layers.BatchNormalization(name=f"{name}_bn")
        self.activate = tf.keras.layers.ReLU(name=f"{name}_relu")
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv1(inputs)
        outputs = self.bn(outputs, training=training)
        outputs = self.activate(outputs)
        outputs = self.dropout(outputs, training=training)
        return outputs
    
    def get_config(self):
        conf = super(TimeDelayLayer, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.bn.get_config())
        conf.update(self.activate.get_config())
        conf.update(self.dropout.get_config())
        return conf


class TimeDelayNNEncoder(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 net_scale: str = "large",
                 dropout: float = 0,
                 name: str = "TDNN_encoder",
                 **kwargs
                 ):
        super(TimeDelayNNEncoder, self).__init__(name=name, **kwargs)
        self.net_scale = net_scale
        if self.net_scale == "super":
            self.tdnn1 = TimeDelayLayer(output_channels=2048, context=3, dilate=1, dropout=dropout, name="tdnn1")
            self.tdnn2 = TimeDelayLayer(output_channels=2048, context=3, dilate=2, dropout=dropout, name="tdnn2")
            self.tdnn3 = TimeDelayLayer(output_channels=2048, context=2, dilate=3, dropout=dropout, name="tdnn3")
            self.tdnn4 = TimeDelayLayer(output_channels=2048, context=2, dilate=3, dropout=dropout, name="tdnn4")
            self.tdnn5 = TimeDelayLayer(output_channels=2048, context=2, dilate=3, dropout=dropout, name="tdnn5")
            self.tdnn6 = TimeDelayLayer(output_channels=2048, context=2, dilate=2, dropout=dropout, name="tdnn6")
            self.tdnn7 = TimeDelayLayer(output_channels=2048, context=6, dilate=1, dropout=dropout, name="tdnn7")
        elif self.net_scale == "large":
            self.tdnn1 = TimeDelayLayer(output_channels=1024, context=3, dilate=1, dropout=dropout, name="tdnn1")
            self.tdnn2 = TimeDelayLayer(output_channels=512, context=3, dilate=2, dropout=dropout, name="tdnn2")
            self.tdnn3 = TimeDelayLayer(output_channels=256, context=2, dilate=3, dropout=dropout, name="tdnn3")
            self.tdnn4 = TimeDelayLayer(output_channels=128, context=2, dilate=3, dropout=dropout, name="tdnn4")
            self.tdnn5 = TimeDelayLayer(output_channels=256, context=2, dilate=3, dropout=dropout, name="tdnn5")
            self.tdnn6 = TimeDelayLayer(output_channels=512, context=2, dilate=2, dropout=dropout, name="tdnn6")
            self.tdnn7 = TimeDelayLayer(output_channels=1024, context=6, dilate=1, dropout=dropout, name="tdnn7")
        elif self.net_scale == "medium":
            self.tdnn1 = TimeDelayLayer(output_channels=128, context=3, dilate=1, dropout=dropout, name="tdnn1")
            self.tdnn2 = TimeDelayLayer(output_channels=64, context=3, dilate=2, dropout=dropout, name="tdnn2")
            self.tdnn3 = TimeDelayLayer(output_channels=40, context=2, dilate=3, dropout=dropout, name="tdnn3")
            self.tdnn4 = TimeDelayLayer(output_channels=40, context=2, dilate=3, dropout=dropout, name="tdnn4")
            self.tdnn5 = TimeDelayLayer(output_channels=40, context=2, dilate=3, dropout=dropout, name="tdnn5")
            self.tdnn6 = TimeDelayLayer(output_channels=64, context=2, dilate=2, dropout=dropout, name="tdnn6")
            self.tdnn7 = TimeDelayLayer(output_channels=128, context=6, dilate=1, dropout=dropout, name="tdnn7")
        elif self.net_scale == "small":
            self.tdnn1 = TimeDelayLayer(output_channels=64, context=3, dilate=1, dropout=dropout, name="tdnn1")
            self.tdnn2 = TimeDelayLayer(output_channels=40, context=3, dilate=3, dropout=dropout, name="tdnn2")
            self.tdnn3 = TimeDelayLayer(output_channels=64, context=15, dilate=1, dropout=dropout, name="tdnn3")
        else:
            raise ValueError("wrong type of net_scale! please choose 'super/large/medium/small'.")
        self.fc = tf.keras.layers.Dense(vocabulary_size)
            
    def call(self, inputs, training=False, mask=None):
        if len(inputs.shape) == 4:
            inputs = tf.squeeze(inputs, axis=-1)
        if self.net_scale in ["super", "large", "medium"]:
            outputs_1 = self.tdnn1(inputs, training=training)
            outputs_2 = self.tdnn2(outputs_1, training=training)
            outputs_3 = self.tdnn3(outputs_2, training=training)
            outputs_4 = self.tdnn4(outputs_3, training=training)
            outputs_5 = self.tdnn5(outputs_4, training=training)
            outputs_6 = self.tdnn6(outputs_5, training=training)
            outputs_7 = self.tdnn7(outputs_6, training=training)
            logits = self.fc(outputs_7, training=training)
        else:
            outputs_1 = self.tdnn1(inputs, training=training)
            outputs_2 = self.tdnn2(outputs_1, training=training)
            outputs_3 = self.tdnn3(outputs_2, training=training)
            logits = self.fc(outputs_3, training=training)
        return logits
    
    def get_config(self):
        conf = super(TimeDelayNNEncoder, self).get_config()
        if self.net_scale in ["super", "large", "medium"]:
            conf.update(self.tdnn1.get_config())
            conf.update(self.tdnn2.get_config())
            conf.update(self.tdnn3.get_config())
            conf.update(self.tdnn4.get_config())
            conf.update(self.tdnn5.get_config())
            conf.update(self.tdnn6.get_config())
            conf.update(self.tdnn7.get_config())
        else:
            conf.update(self.tdnn1.get_config())
            conf.update(self.tdnn2.get_config())
            conf.update(self.tdnn3.get_config())
        conf.update(self.fc.get_config())
        return conf


class TimeDelayNNTransducer(Transducer):
    def __init__(self,
                 vocabulary_size: int,
                 net_scale: str = "large",
                 dropout: float = 0,
                 embed_dim: int = 512,
                 embed_dropout: int = 0,
                 num_lstms: int = 1,
                 lstm_units: int = 512,
                 joint_dim: int = 1024,
                 name: str = "TDNN_transducer",
                 speech_config=dict,
                 **kwargs
                 ):
        super(TimeDelayNNTransducer, self).__init__(
            encoder=TimeDelayNNEncoder(
                vocabulary_size=vocabulary_size,
                net_scale=net_scale,
                dropout=dropout
            ),
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_lstms=num_lstms,
            lstm_units=lstm_units,
            joint_dim=joint_dim,
            name=name,
            speech_config=speech_config,
            **kwargs
        )
        self.time_reduction_factor = 1


if __name__ == "__main__":
    from utils.user_config import UserConfig
    from utils.text_featurizers import TextFeaturizer
    from utils.speech_featurizers import SpeechFeaturizer
    import time

    config = UserConfig(r'../configs/am_data_back.yml', r'../configs/tdnn.yml')
    config['decoder_config'].update({'model_type': 'Transducer'})
    config['speech_config'].update({'use_mel_layer': False})

    Tfer = TextFeaturizer(config['decoder_config'])
    SFer = SpeechFeaturizer(config['speech_config'])
    f, c = SFer.compute_feature_dim()
    config['model_config'].update({'vocabulary_size': Tfer.num_classes})
    # config['model_config'].update({'startid': Tfer.start})
    config['model_config'].pop('LAS_decoder')
    config['model_config'].pop('enable_tflite_convertible')

    tdnnt = TimeDelayNNTransducer(**config['model_config'],
                                  speech_config=config['speech_config'])
    # ct.add_featurizers(Tfer)
    x = tf.ones([1, 300, f, c])
    length = tf.constant([300])
    out = tdnnt._build([1, 300, f, c])
    print(out.shape)
    print(1)

