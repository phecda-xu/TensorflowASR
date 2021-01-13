# coding: utf-8

import numpy as np
import tensorflow as tf

from AMmodel.model import AM
from utils.user_config import UserConfig
from utils.text_featurizers import TextFeaturizer
from utils.speech_featurizers import SpeechFeaturizer


class StreamingASR(object):
    def __init__(self, config):
        self.am = AM(config)
        self.am.load_model(False)
        self.speech_config = config['speech_config']
        self.text_config = config['decoder_config']
        self.speech_feature = SpeechFeaturizer(self.speech_config)
        self.text_featurizer = TextFeaturizer(self.text_config)
        self.decoded = tf.constant([self.text_featurizer.start])

    def stream_detect(self, inputs):
        data = self.speech_feature.load_wav(inputs)

        if self.am.model.mel_layer is None:
            mel = self.speech_feature.extract(data)
            x = np.expand_dims(mel, 0)
        else:
            mel = data.reshape([1, -1, 1])
            x = self.am.model.mel_layer(mel)
        x = self.am.model.encoder(x)   # TensorShape([1, 109, 144])
        step = x.shape[1]
        i = 0
        while i < step:
            self.step_decode(x[:, i])
            i = i+1

    def step_decode(self, step_input):
        enc = tf.reshape(step_input, [1, 1, -1])
        y = self.am.model.predict_net(inputs=tf.reshape(self.decoded, [1, -1]),
                                      p_memory_states=None,
                                      training=False)
        y = y[:, -1:]
        z = self.am.model.joint_net([enc, y], training=False)
        probs = tf.squeeze(tf.nn.log_softmax(z))
        pred = tf.argmax(probs, axis=-1, output_type=tf.int32)
        pred = tf.reshape(pred, [1])
        if pred != 0 and pred != self.text_featurizer.blank:
            self.decoded = tf.concat([self.decoded, pred], axis=0)
            print("pred: {}".format(self.text_featurizer.index_to_token[pred.numpy().tolist()[0]]))

    def predict_stack_buffer(self, wavfile):
        data = self.speech_feature.load_wav(wavfile)
        buffer_step = int(len(data) / 16000)
        j = 0
        while j < buffer_step:
            buffer = data[j * 16000 - j * 5000: (j + 1) * 16000]
            if self.am.model.mel_layer is None:
                mel = self.speech_feature.extract(buffer)
                x = np.expand_dims(mel, 0)
            else:
                mel = buffer.reshape([1, -1, 1])
                x = self.am.model.mel_layer(mel)
            x = self.am.model.encoder(x)
            step = x.shape[1]
            i = 0
            while i < step:
                enc = tf.reshape(x[:, i], [1, 1, -1])
                y = self.am.model.predict_net(inputs=tf.reshape(self.decoded, [1, -1]),
                                              p_memory_states=None,
                                              training=False)
                y = y[:, -1:]
                z = self.am.model.joint_net([enc, y], training=False)
                logits = tf.squeeze(tf.nn.log_softmax(z))
                pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
                pred = tf.reshape(pred, [1])
                if pred != 0 and pred != self.text_featurizer.blank:
                    self.decoded = tf.concat([self.decoded, pred], axis=0)
                    print("buffer_step: {}, "
                          "step: {}, "
                          "pred: {}".format(j,
                                            i,
                                            self.text_featurizer.index_to_token[pred.numpy().tolist()[0]]))
                i += 1
            j += 1
        print(1)


if __name__ == "__main__":
    am_config = UserConfig(r'./pre_train/rnnt/am_data.yml', r'./pre_train/rnnt/conformer.yml')
    # am_config = UserConfig(r'./configs/am_data_back.yml', r'./configs/tdnn.yml')
    model = StreamingASR(am_config)
    model.stream_detect('CppInference/test.wav')



