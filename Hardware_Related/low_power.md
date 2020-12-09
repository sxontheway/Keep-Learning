# 用 GPIO pin 唤醒 Jetson Nano
> Jetson 系列 Q&A：https://elinux.org/Jetson/FAQ/BSP

## 目标
在大多数时间，让nano处于低能耗的suspend模式。当有外界一个电平改变时（例如PIR sensor），nano被唤醒，执行相应任务（例如深度学习）。这样即便用电池供电，也能有较长的续航时间

---
<br>


## 背景知识
### Linux 电源管理
* 一般有四种状态，能耗从大到小：
    * freeze
    * standby
    * mem (Suspend to Memory, also called sleep) 
    * disk (Suspend to Disk, also called hibernation)  

* TX2 上只支持 freeze，mem，可用 `cat /sys/power/state` 查看 
    https://forums.developer.nvidia.com/t/suspend-to-ram-and-hibernate-in-jetson-tx2-using-psci/52724  

* Nano 原系统只支持 freeze，但经过一系列操作后可支持 mem

### Linux DTS (Device Tree Source) 介绍 
> https://e-mailky.github.io/2019-01-14-dts-1  
> https://blog.csdn.net/qq_16777851/article/details/87291146   

[Embedded Device 为什么要用设备树，PC需要吗？](https://unix.stackexchange.com/questions/399619/why-do-embedded-systems-need-device-tree-while-pcs-dont) 设备树，可以帮助我们在修改了元器件需要用不同的 driver 时，通过修改几行配置得到新的 dts, 然后生成新的 dtb 文件替换老的，而不用重新交叉编译整个kernel

`*.dts`： Device Tree Source，是一种ASCII文本对device tree的描述。在ARM Linux中，一个`.dts`文件对应一个ARM的machine  (一个SoC可能对应多个machine，也即一个SoC可以对应多个产品和电路板），源码一般放在内核的`/arch/arm64/boot/dts`目录

`*.dtsi` 文件作用：由于一个SOC可能有多个不同的电路板，而每个电路板拥有一个 `*.dts`，这些dts势必会存在许多共同部分（因为他们共用一个SOC），为了减少代码的冗余，设备树将这些共同部分提炼保存在 `*.dtsi` 文件中，供不同的dts共同使用。`*.dtsi` 的使用方法，类似于C语言的头文件，在dts文件中需要进行 `include *.dtsi` 文件。当然，dtsi本身也支持include另一个dtsi文件

`*.dtb` 文件：由`.dts`，`.dtsi`编译形成的二进制文件，一般可以在`/boot/dtb`找到

---

### Nano 的几个硬件版本
我的是 a02 (核心板) + devkit (无emmc，系统全部运行在sd卡上)，参见：
https://blog.csdn.net/u013673476/article/details/104794955  
https://docs.nvidia.com/sdk-manager/system-requirements/index.html  
所以 .dts, .dtb相关文件前缀应该是 `tegra210-p3448-0000-p3449-0000-a02`，flash时的.conf配置文件应该是：`jetson-nano-qspi-sd.conf`


---

### Nano devkit 的三种刷机方法
> Nano devkit recovery 模式，要用跳线帽：https://forums.developer.nvidia.com/t/how-to-put-nvidia-jetson-nano-in-force-recovery-mode/80400  

三种方法：
*  像树莓派一样用sd卡做系统盘
*  用 sdk manager，分三步  
    * 下载源码（要和设备对应上）并在host编译安装
        有时在 sdk manager 中选择下载最新版本Jetpack会报错，应该是nvidia的锅，还没把文件挂上网；其中 step01 可以不选 host，只选 target  
    * flash到nano（可在 sdkmanager 中选择只下载不安装，略去该步骤，然后手动用命令行 flash） 

* 全部用命令行完成
    > [Flashing and Booting the Target Device](
    https://docs.nvidia.com/jetson/archives/l4t-archived/l4t-322/index.html#page/Tegra%2520Linux%2520Driver%2520Package%2520Development%2520Guide%2Fflashing.html%23)   

    * 下载源码
    * 编译 kernel，modules，dtb...
    * flash   
    在 `nvidia-sdk/JetPack_<version>/Linux_for_Tegra` 目录下： `sudo ./flash.sh <platform> <rootdev> ` 例如，flash整个kernel：`sudo ./flash.sh jetson-nano-qspi-sd mmcblk0p1`


---
<br>

## 使能 Nano 的 suspend 模式
一些版本的 Nano 刷机后只支持 freeze，但经过一系列操作后可支持 mem（也即 suspend），见： https://forums.developer.nvidia.com/t/how-to-suspend-on-jetson-nano/79048  

1. 需要在host的 `nvidia-sdk/JetPack_<version>/Linux_for_Tegra` 安装文件夹中，找到 `*.conf 文件`（例如对于我的a02 devkit，是 `jetson-nano-qspi-sd.conf`），加上 `ODMDATA=0x94800;` 
2. 然后再 flash（根据自己的板子和、要flash的文件，选则不同命令）


---
<br>

## 用 GPIO 唤醒 nano（通过更新 .dtb 实现）
1. 通过 sdkmanager 下载源码（它会自动帮你在 host 上编译好）

1.  下载 .dts 源码  
例如使用 `./source_sync.sh -k tegra-l4t-r32.2.0`，注意L4T版本号要对的上  
    * [Where do we find these other DTSI files?](https://forums.developer.nvidia.com/t/where-do-we-find-these-other-dtsi-files/77905)
    * [Why it is not download file after I run “./source_sync.sh”？](https://forums.developer.nvidia.com/t/why-it-is-not-download-file-after-i-run-source-sync-sh-at-tx2-r32-2/81266)

1. 修改 .dts 文件
    * Nano   
    注意自己用的nano版本，例如我的是 a02 devkit，前缀应该为 `tegra210-p3448-0000-p3449-0000-a02`（见上文对nano硬件版本的介绍），步骤见：
    https://elinux.org/Jetson/FAQ/BSP/Nano_Wakeup_Pin
    * TX1   
    https://forums.developer.nvidia.com/t/configuring-gpio-wake/59912/5

1. 编译 .dtb 文件
    > [How to build NVIDIA Jetson Nano kernel](https://developer.ridgerun.com/wiki/index.php?title=Jetson_Nano/Development/Building_the_Kernel_from_Source)  

    主要步骤：
    * Download and install the Toolchain
    * Download the kernel sources (这一步在上文中下载 .dts 那一步已经做了)
    * Compile kernel and dtb (可以只选 dtb)  
    其中的 `make dtbs` 编译在底层也是调用了 device tree compiler (dtc)： https://stackoverflow.com/questions/21670967/how-to-compile-dts-linux-device-tree-source-files-to-dtb   
    * 用新编译得到的 `.dtb` 替换 `Linux_for_Tegra/kernel/dtb` 中的 `.dtb`。需要找到自己板子对应的版本，例如对于我的板子，是 `tegra210-p3448-0000-p3449-0000-a02.dtb`

1. flash  
    > [How to build NVIDIA Jetson Nano kernel](https://developer.ridgerun.com/wiki/index.php?title=Jetson_Nano/Development/Building_the_Kernel_from_Source)

    只 flash dtb： `sudo ./flash.sh -r -k DTB jetson-nano-qspi-sd mmcblk0p1
`

---
<br>

## 其他
### 进入 Suspend
* 查看支持的模式：`cat /sys/power/state`  
* 进入suspend：`systemctl suspend `

### Jetson Pin Mapping
* TX2:  https://www.jetsonhacks.com/nvidia-jetson-tx2-j21-header-pinout/  
* Nano: https://www.jetsonhacks.com/nvidia-jetson-nano-j41-header-pinout/  
* TX2: https://docs.google.com/spreadsheets/d/18o5R4xghozzdU1fbAtjox0kI5yaGx96U73s6xsmcKLM/edit#gid=409792609
 
### .dts 文件理解
``` dts
gpio-keys {
    wake_up {
        label = "Wake_up";	// 随便写

        // gpio 标识符由3个数字来识别：gpio的组, 组内的gpio口序号, gpio口的有效状态
        // 见 https://forums.developer.nvidia.com/t/configuring-gpio-wake/59912/2
        gpios = <&gpio TEGRA_GPIO(G, 3) GPIO_ACTIVE_HIGH>;
        
        // 对于Linux中，每一个event都有唯一的编码，
        // 见 https://elixir.bootlin.com/linux/latest/source/include/uapi/linux/input-event-codes.h 
        linux,code = <KEY_WAKEUP>; 

        // Indicate that this key can wake up the system
        gpio-key,wakeup;

        // 在 https://developer.nvidia.com/zh-cn/embedded/downloads 中找到 pinmux table 
        // table 中 Column Y (Wake) 每个 pin 对应不同的 wake up ID 
        nvidia,pmc-wakeup = <&tegra_pmc PMC_WAKE_TYPE_GPIO 6 PMC_TRIGGER_TYPE_NONE>
        
        // 定义该 key 的去抖间隔 ms
        debounce-interval = <30>;  
    };
};
```