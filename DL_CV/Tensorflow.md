# 1.安装
# 1.1 后端Tensorflow的安装
按照官网用pip安装： https://www.tensorflow.org/install/pip?hl=zh-cn  
* Warning： Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
  * 原因：  
  当用GPU时，可忽略它。 这个警告简而言之就是CPU支持AVX（高级向量扩展指令集），但是安装的tensorflow并没有运用这个功能，不能实现CPU上的加速，参见： 
  https://blog.csdn.net/hq86937375/article/details/79696023
  * 解决方案：  
  从源码安装可解决这个问题，参见：https://github.com/lakshayg/tensorflow-build

# 1.2 前端Kares的安装
* 为配合tensorflow， 使用python3: 
  * 进入python3的虚拟环境(venv为创建虚拟环境时自定义的文件夹名称):` source ./venv/bin/activate`
  * `pip install keras`
