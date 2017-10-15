#RNN

##1
model_rnn.py is built based on this [blog](https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html)

##2 細節
關於架構的部分，如果用"真正的recurrent":無論多長,循環來計算.在feed-forward的時候沒有問題，但是

>There’s a problem with using this approach for training: the gradients computed during backpropagation are graph-bound

解決的方法是,

>Alternatively, we might make our graph as wide as our data sequence

也就是說我們規定我們的graph最多接受多長的sequence(這樣的展開就不是無限定長度),但是這樣也有問題

>In the words of Pascanu et al., “in the same way a product of [k] real numbers can shrink to zero or explode to infinity, so does this product of matrices …”

即梯度消失. 解決這個問題常用的方法是truncate我們bp的過程(truncated-BPTT):

> We backpropagate every possible error n steps
> That is, if we have a sequence of length 49, and choose n=7, we would backpropagate 42 of the errors the full 7 steps

在tensorflow中的做法就是

>We would take our sequence of length 49, break it up into 7 sub-sequences of length 7 that we feed into the graph in 7 separate computations, and that only the errors from the 7th input in each graph are backpropagated the full 7 steps

關於BPTT的部分,我們將RNN展開如下圖,

<p align="center">
<img src="https://r2rt.com/static/images/BasicRNNLabeled.png">
</p>

實作上,展開可以用`tf.get_variable()`來share.這邊code中的`state_size`指的是`s_t`的維度.例如`state_size=4`意味著`X_t`連接到了4個neurons

##3 測試資料
輸入 X_t 是隨機的0/1 (0.5的伯努利)

正確輸出 Y_t 是這樣定義的：參考 X_{t-3}和X_{t-8},概率如下表

| X_{t-3} | X_{t-8} | P(Y_t=1) |
|:-------:|:-------:|:-------:|
|    0    |    0    |   0.5   |
|    0    |    1    |   0.25  |
|    1    |    0    |    1    |
|    1    |    1    |   0.75  |

所以如果model沒有學到dependencies,會預測有0.25*(0.5+0.25+1+0.75)=0.625的幾率是1

所以如果model學到X_{t-3},會預測有0.25*(0.5+0.25+1+0.75)=0.625的幾率是1

所以如果model沒有學到dependencies,會預測有0.25*(0.5+0.25+1+0.75)=0.625的幾率是1