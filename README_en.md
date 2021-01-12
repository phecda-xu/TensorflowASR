<h1 align="center">
<p>TensorflowASR</p>
<p align="center">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.2.0-orange">
</p>
</h1>
<h2 align="center">
<p>State-of-the-art Automatic Speech Recognition in Tensorflow 2</p>
</h2>
<p align="center">
CTC\Transducer\LAS Default is Chinese ASR
</p>
<p align="center">
Now the project is still in the development stages
</p>
<p align="center">
Welcome to use and feedback bugs

English|[中文版](https://github.com/Z-yq/TensorflowASR/)
</p>



## Mel Layer

Provides a feature extraction layer using tensorflow to reference librosa for end-to-end integration with other platforms.

Using it:
- am_data.yml 
   ```
   use_mel_layer: True
   mel_layer_type: Melspectrogram #Spectrogram
   trainable_kernel: True #support train model,not recommend
   ```

## Cpp Inference
A call example for C++ is provided.

Demo for TensorflowC 2.3.0

detail in [cppinference](https://github.com/Z-yq/TensorflowASR/tree/master/CppInference)
## Pretrained Model

All test on _`AISHELL TEST`_ datasets.

**AM:**

Model Name|Mel layer(USE/TRAIN)| link                                          |code|train data        |txt CER|phoneme CER|Model Size|
----------|--------------------|-----------------------------------------------|----|------------------|-------|-----------|---------|
MultiTask |False/False|pan.baidu.com/s/1nDDqcJXBbpFJASYz_U8FfA        |ucqf|aishell2(10 epochs)|10.4   |8.3        |109M|
ConformerRNNT(S)|True/True|pan.baidu.com/s/1bdqeLDBHQ_XmgNuUr6mflw|fqvf|aishell2(10 epochs)|-|9.7|61M|
ConformerCTC(S)|True/True|pan.baidu.com/s/1sh2bUm1HciE6Fu7PHUfRGA|jntv|aishell2(10 epochs)|-|9.9|46M|
ConformerCTC2(S)|True/False|pan.baidu.com/s/12hsjq-lWudeaQzQomV-PDw|ifm6|aishell2(10 epochs)|-|8.1|46M|
ConformerCTC3(S)|False/False|pan.baidu.com/s/1zKDgMHfpOhw10pOSWmtLrQ|gmr5|aishell2(10 epochs)|-|7.0|46M|

**LM:**

Model Name|O2O(Decoder)| link |code|train data|txt cer|model size|params size|
---------|----|------|----|-------|------|----------|-----------|
TransformerO2OE|True(False)|pan.baidu.com/s/1lyqHGacYd7arBrJtlTFdTw|kw0y|aishell2 text(98k steps)|4.4|200M|52M|
TransformerO2OED|True(True)|pan.baidu.com/s/1acvCRpS2j16dxLoCyToB6A|jrfi|aishell2 text(10k steps)|6.2|217M|61M|
Transformer|True(True)|pan.baidu.com/s/1W3HLNNGL3ceJfoxb0P7RMw|qeet|aishell2 text(10k steps)|8.6|233M|61M|
TransformerPunc|False(True)|pan.baidu.com/s/1umwMP2nIzr25NnvG3LTRvw|7ctd|translation task texts|-|76M|30M|
**Speed:**

AM Speed Test(Python), a ~4.1 seconds wav on **CPU**:

|CTC    |Transducer|LAS  |
|-------|----------|-----|
|150ms  |350ms     |280ms|

LM Speed Test(Python),12 words on **CPU**:

|O2O-Encoder-Decoder|O2O-Encoder|Encoder-Decoder|
|-------------------|-----------|---------------|
|              100ms|       20ms|          300ms|


## Community
welcome to join 

<img width="300" height="300" src="./community.jpg">


## What's New?

New:

- Change RNNT predict to support C++
- Add C++ Inference Demo,detail in [cppinference](https://github.com/Z-yq/TensorflowASR/tree/master/CppInference)
    

## Supported Structure
-  **CTC**
-  **Transducer**
-  **LAS**
-  **MultiTaskCTC**

## Supported Models

-   **Conformer** 
-   **ESPNet**:`Efficient Spatial Pyramid of Dilated Convolutions`
-   **DeepSpeech2**
-   **Transformer**` Pinyin to Chinese characters` 
       -  O2O-Encoder-Decoder `Complete transformer,and one to one relationship between phoneme and target
,e.g.: pin4 yin4-> 拼音`
       -  O2O-Encoder `Not contain the decoder part,others are same.`
       -  Encoder-Decoder `Typic transformer`


## Requirements

-   Python 3.6+
-   Tensorflow 2.2+: `pip install tensorflow`
-   librosa
-   pypinyin `if you need use the default phoneme`
-   keras-bert
-   addons `For LAS structure,pip install tensorflow-addons`
-   tqdm
-   jieba
-   wrap_rnnt_loss `not essential,provide in ./externals`
-   wrap_ctc_decoders `not essential,provide in ./externals`

## Usage

1. Prepare train_list.

    **am_train_list** format:

    ```text
    file_path1 \t text1
    file_path2 \t text2
    ……
    ```

    **lm_train_list** format:
    ```text
    text1
    text2
    ……
    ```
2. Down the bert model for LM training,if you don't need LM can skip this Step:
            
        https://pan.baidu.com/s/1_HDAhfGZfNhXS-cYoLQucA extraction code: 4hsa
        
3. Modify the **_`am_data.yml`_** (in ./configs),set running params.Modify the `name` in **model yaml** to choose the structure.
4. Just run:
  
     ```shell
    python train_am.py --data_config ./configs/am_data.yml --model_config ./configs/conformer.yml
    ```
  
5. To Test,you can follow in **_`run-test.py`_**,addition,you can modify the **_`predict`_** function to meet your needs:
     ```python
    from utils.user_config import UserConfig
    from AMmodel.model import AM
    from LMmodel.trm_lm import LM
    
    am_config=UserConfig(r'./configs/am_data.yml',r'./configs/conformer.yml')
    lm_config = UserConfig(r'./configs/lm_data.yml', r'./configs/transformer.yml')
    
    am=AM(am_config)
    am.load_model(training=False)
    
    lm=LM(lm_config)
    lm.load_model()
    
    am_result=am.predict(wav_path)
    if am.model_type=='Transducer':
        am_result =am.decode(am_result[1:-1])
        lm_result = lm.predict(am_result)
        lm_result = lm.decode(lm_result[0].numpy(), self.lm.word_featurizer)
    else:
        am_result=am.decode(am_result[0])
        lm_result=lm.predict(am_result)
        lm_result = lm.decode(lm_result[0].numpy(), self.lm.word_featurizer)
   
    ```
Use **Tester** to test your model:
Fisrt modify the `eval_list`  in _`am_data.yml/lm_data.yml`_

Then:
```shell
python eval_am.py --data_config ./configs/am_data.yml --model_config ./configs/conformer.yml
```
Tester will show **SER/CER/DEL/INS/SUB** 

## Your Model

You can add your model in `./AMmodel` folder e.g, LM model is the same with follow:

```python

from AMmodel.transducer_wrap import Transducer
from AMmodel.ctc_wrap import CtcModel
from AMmodel.las_wrap import LAS,LASConfig
class YourModel(tf.keras.Model):
    def __init__(self,……):
        super(YourModel, self).__init__(……)
        ……
    
    def call(self, inputs, training=False, **kwargs):
       
        ……
        return decoded_feature
        
#To CTC
class YourModelCTC(CtcModel):
    def __init__(self,
                ……
                 **kwargs):
        super(YourModelCTC, self).__init__(
        encoder=YourModel(……),num_classes=vocabulary_size,name=name,
        )
        self.time_reduction_factor = reduction_factor #if you never use the downsample layer,set 1

#To Transducer
class YourModelTransducer(Transducer):
    def __init__(self,
                ……
                 **kwargs):
        super(YourModelTransducer, self).__init__(
            encoder=YourModel(……),
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_lstms=num_lstms,
            lstm_units=lstm_units,
            joint_dim=joint_dim,
            name=name, **kwargs
        )
        self.time_reduction_factor = reduction_factor #if you never use the downsample layer,set 1

#To LAS
class YourModelLAS(LAS):
    def __init__(self,
                ……,
                config,# the config dict in model yml
                training,
                 **kwargs):
        config['LAS_decoder'].update({'encoder_dim':encoder_dim})# encoder_dim is your encoder's last dimension
        decoder_config=LASConfig(**config['LAS_decoder'])

        super(YourModelLAS, self).__init__(
        encoder=YourModel(……),
        config=decoder_config,
        training=training,
        )
        self.time_reduction_factor = reduction_factor #if you never use the downsample layer,set 1

```
Then,import the your model in `./AMmodel/model.py` ,modify the `load_model` function
## Convert to pb
AM/LM model are the same as follow:
```python
from AMmodel.model import AM
am_config = UserConfig('...','...')
am=AM(am_config)
am.load_model(False)
am.convert_to_pb(export_path)
```
## Tips
IF you want to use your own phoneme,modify the convert function in `am_dataloader.py/lm_dataloader.py`

```python
def init_text_to_vocab(self):#keep the name
    
    def text_to_vocab_func(txt):
        return your_convert_function

    self.text_to_vocab = text_to_vocab_func #here self.text_to_vocab is a function,not a call
```

Don't forget that the token list start with **_`S`_** and **_`/S`_**,e.g:

        S
        /S
        de
        shì
        ……




## References

Thanks for follows:


https://github.com/usimarit/TiramisuASR `modify from it`

https://github.com/noahchalifour/warp-transducer

https://github.com/PaddlePaddle/DeepSpeech

https://github.com/baidu-research/warp-ctc

## Licence

允许并感谢您使用本项目进行学术研究、商业产品生产等，但禁止将本项目作为商品进行交易。

Overall, Almost models here are licensed under the Apache 2.0 for all countries in the world.

Allow and thank you for using this project for academic research, commercial product production, allowing unrestricted commercial and non-commercial use alike. 

However, it is prohibited to trade this project as a commodity.