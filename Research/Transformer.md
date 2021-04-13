> https://zhuanlan.zhihu.com/p/80986272

* tranformer 优于 LSTM/RNN 的点
    * 数据并行的问题，一次可以输入多个单词，而不像 LSTM/RNN 需要一个接一个
    * transfer learning的问题，LSTM 几乎不支持 transfer learning，transformer可以
* 注意：tranformer 中，encoder可以并行计算，一次性全部encoding出来，但decoder不是一次把所有序列解出来的，而是像rnn一样一个一个解出来的，因为要用上一个位置的输入当作attention的query
* LSTM is still good when:
    * sequence too long. The complexity of transformer is O(N^2)
    * 数据集本身比较小，transform要训练好所需要的数据量比较大