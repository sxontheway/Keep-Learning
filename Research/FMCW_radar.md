## 为什么 FFT 之后要 fftshift?
> https://www.itread01.com/content/1541093349.html

fftshift 将零频点从四个角上移到频谱的中间，方便观察，主要利用的是平移不变性。

原因：离散信号的频谱是有周期性，采样点数为N的DTF变换，频谱的周期为N。FFT得到的频谱，在`0`、`f_s`、`2f_s` 都有较大的频率响应（f_s是采样频率）。因为 `(-f_s/2, 0)` 和 `(f_s/2, f_s)` 是一样的，所以fftshift操作并改变什么（DFT得到的一个连绵无限的周期性频谱图，我们显示的fft结果不过截取了其中一部分，fftshift本质上是将我们截取的这部分的窗口移动了一下，使得零频谱移到中央）

---
<br>

## 离散傅立叶变换 DFT
> https://zh.wikipedia.org/zh-hans/%E7%A6%BB%E6%95%A3%E5%82%85%E9%87%8C%E5%8F%B6%E5%8F%98%E6%8D%A2  
  <p align="center" >
	<img src="./pictures/dft.png" width="800">
  </p>

* 傅立叶变换的输出是复数吗？  
是的，求幅值可取绝对值，求相位可用`arctan(-real/image)`，一般是有一个求幅值的操作（因为要找到幅值最大的频率，然后该频率在FMCW下对应的距离、速度等）

* 为什么输出是复数？  
`x[n]`是实数，`x_hat[k]`明显是复数 
* 为什么 FFT 代表从时域到频域的变换？  
IDFT中，`x_hat[k]` 乘以不同频率的正弦/余弦信号，得到原信号 `x[n]`，所以可理解为 `x_hat[k]` 就是这些不同频率的正弦/余弦信号的幅值，也即频谱图（f-A）

---
<br>

## Range FFT, Doppler FFT 和 AoA Estimation
> https://blog.csdn.net/nuaahz/article/details/90719605 
* Range FFT是对一个chirp上的ADC采样进行的。range FFT将时域信号(坐标是t, A) 变为频域信号(坐标是f, A)，A是幅度，因为 FMCW 中 IF 信号频率 f 正比于距离，所以用 FFT 可以求出距离  
* Dopper FFT 是对一个 frame 内不用 chirp 的同一 range bin 的数据进行的。因为频率基本相同，doppler FFT 关注的是 IF 信号相位变化。doppler FFT 采样的时间间隔是 chirp interval。phase changing 信号的频率和两个 chirp 之间的相位差有关，而相位差和速度成正比。所以求速度转化为求 phase signal 的频率，显然 FFT 可以派上用场。

  <p align="center" >
	<img src="./pictures/dopplerfft.png" width="800">
  </p>

* AOA Estimation 也可以用 FFT，原理是 virtual antennas 接受到的 IF 信号的频率差值正比于距离差值，而距离差值正比于 `sin(\theta)`。所以 `\theta` 可以通过对不同  virtual antennas 的 IF 信号进行 FFT 得到。

* FFT 怎么和物理世界结合起来？
因为FFT变换得到频域，FMCW中，由频率可推出距离、速度、角度，所以可以测距、测速、测角度。
DFT中，k前面的系数是`(2*pi*n/N)`，也即一个bin对应的分辨率。一共N个bin，一个bin对应的分辨率乘以 N 就是最大量程。FFT 得出 bin_index 后，乘以一个分辨率，即得到物理世界中的数值（距离、速度）

---
<br>

## FMCW processing pipeline
> https://e2echina.ti.com/question_answer/analog/other_analog/f/60/p/197625/614435

* Pipeline 1：range -> angle -> doppler
    * 适合近距离高分辨率应用、生成 range-azimuth map
    * 因为 angle processing 在 doppler fft 之前，所以可利用 chirp 间的复数信息构建高质量协方差矩阵，这有利于超分辨率算法（Capon, Music等）
* Pipeline 2: range -> doppler -> angle
    * 适合物体检测、点云估计、生成 range-doppler map
    * 因为只能利用 doppler FFT 之后的 single snapshot 数据进行角度估计。
 
当然也可以双分支，range-fft 之后用两个分支分别进行 doppler FFT 和 AoA Estimation，例如 mmWave industrial toolbox 中的 3D people counting。
   
---
<br>

## 超分辨率 AOA 算法
> [基于Bartlett算法和Capon算法的DOA估计](https://wenku.baidu.com/view/3bde1b58cfc789eb162dc854.html)  
> [阵列信号处理中DOA算法分类总结](https://wenku.baidu.com/view/5d9d869a0912a2161579299b.html)

总的来说分辨率上，MUSIC > Capon > Bartlett > FFT
* 延迟相加法/Bartlett法  
目标函数是使得来自信号方向的功率最大

* Capon 最小方差法  
约束条件是使得来自期望方向的信号功率不变；目标函数使得输出总功率最小，会用到原信号的协方差矩阵

* 子空间法：MUSIC，ESPRIT  
将信号分成两个子空间（信号空间、噪音空间）

