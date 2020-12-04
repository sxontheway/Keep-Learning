## Linux 电源管理
* 一般有四种状态，能耗从大到小：
    * freeze
    * standby
    * mem (Suspend to Memory, also called sleep) 
    * disk (Suspend to Disk, also called hibernation)  

    TX2 上只支持 freeze，mem，可用 `cat /sys/power/state` 验证
    https://forums.developer.nvidia.com/t/suspend-to-ram-and-hibernate-in-jetson-tx2-using-psci/52724  

* 休眠与唤醒 
    > https://cloud.tencent.com/developer/article/1350774


## 用GPIO唤醒唤醒Jetson
### 背景
* Linux DTS (Device Tree Source) 介绍 
    > https://e-mailky.github.io/2019-01-14-dts-1#dts%E7%9A%84%E5%8A%A0%E8%BD%BD%E8%BF%87%E7%A8%8B  
    > https://blog.csdn.net/qq_16777851/article/details/87291146 

    简单来说用 DTS 的原因：如果改了元器件，需要用不同的 driver 时，可以通过修改几行配置生成新的 dts, 然后生成替换 uboot 中的 dts 文件, 1分钟后板子就能用了，不像以前还要改 makefile，编译烧写，很麻烦

    `*.dts` 文件是一种ASCII文本对Device Tree的描述。在ARM Linux在，一个`.dts`文件对应一个ARM的machine，一般放置在内核的的`/arch/arm/boot/dts`目录。一般而言，一个 `*.dts` 文件对应一个ARM的machine (一个SoC可能对应多个machine，也即一个SoC可以对应多个产品和电路板）

    `*.dtsi` 文件作用：由于一个SOC可能有多个不同的电路板，而每个电路板拥有一个 `*.dts`，这些dts势必会存在许多共同部分（因为他们共用一个SOC），为了减少代码的冗余，设备树将这些共同部分提炼保存在 `*.dtsi` 文件中，供不同的dts共同使用。`*.dtsi` 的使用方法，类似于C语言的头文件，在dts文件中需要进行 `include *.dtsi` 文件。当然，dtsi本身也支持include另一个dtsi文件。

* Config wake up pin on Jetson Nano
    > https://elinux.org/Jetson/FAQ/BSP