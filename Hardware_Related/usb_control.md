## /sys/bus/usb/devices 目录下的命名

```
alex@alex-nano:/sys/bus/usb/devices$ lsusb
Bus 002 Device 002: ID 0bda:0411 Realtek Semiconductor Corp. 
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 034: ID 2341:0043 Arduino SA Uno R3 (CDC ACM)
Bus 001 Device 038: ID 0bda:8152 Realtek Semiconductor Corp. 
Bus 001 Device 037: ID 03f0:2f4a Hewlett-Packard 
Bus 001 Device 036: ID 03f0:094a Hewlett-Packard Optical Mouse [672662-001]
Bus 001 Device 035: ID 1a40:0101 Terminus Technology Inc. Hub
Bus 001 Device 002: ID 0bda:5411 Realtek Semiconductor Corp. 
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub

alex@alex-nano:/sys/bus/usb/devices$ lsusb -t
/:  Bus 02.Port 1: Dev 1, Class=root_hub, Driver=tegra-xusb/4p, 5000M
    |__ Port 1: Dev 2, If 0, Class=Hub, Driver=hub/4p, 5000M
/:  Bus 01.Port 1: Dev 1, Class=root_hub, Driver=tegra-xusb/5p, 480M
    |__ Port 2: Dev 2, If 0, Class=Hub, Driver=hub/4p, 480M
        |__ Port 1: Dev 35, If 0, Class=Hub, Driver=hub/4p, 480M
            |__ Port 2: Dev 36, If 0, Class=Human Interface Device, Driver=usbhid, 1.5M
            |__ Port 3: Dev 37, If 0, Class=Human Interface Device, Driver=usbhid, 1.5M
            |__ Port 3: Dev 37, If 1, Class=Human Interface Device, Driver=usbhid, 1.5M
            |__ Port 4: Dev 38, If 0, Class=Vendor Specific Class, Driver=r8152, 480M
        |__ Port 4: Dev 34, If 1, Class=CDC Data, Driver=cdc_acm, 12M
        |__ Port 4: Dev 34, If 0, Class=Communications, Driver=cdc_acm, 12M

alex@alex-nano:/sys/bus/usb/devices$ ls
1-0:1.0  1-2.1    1-2.1:1.0  1-2.1.2:1.0  1-2.1.3:1.0  1-2.1.4      1-2.4      1-2.4:1.1  2-1      usb1
1-2      1-2:1.0  1-2.1.2    1-2.1.3      1-2.1.3:1.1  1-2.1.4:1.0  1-2.4:1.0  2-0:1.0    2-1:1.0  usb2
```

基本规则：`bus-port:configuration.interface`  
例如 `1-2.1.3:1.1` 表示： bus1 -> port2 -> port1 -> port3 对应的设备（是 device 37，是个键盘），配置号1 (If 1)，对应接口编号为 1
