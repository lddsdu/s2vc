## S2VC
sequence-to-sequence video caption

这个model主要参考了 sequence to sequence - video to text 中的video caption 的方法

- data
    - 每个video抽取80 frames, 
    - 然后使用预训练的vgg-16获得特征 80 * 4096

- model
    - encoder-decoder
    - attention
   
