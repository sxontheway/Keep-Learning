* Firefox cannot synchronize  
The add-on manager is not installed: https://blog.csdn.net/JohnYu024/article/details/79763300  
Add-on manager link: http://mozilla.com.cn/thread-343905-1-1.html

* No Wifi Model  
`iwconfig`: see whether there is a model called wlan0. If not, it means that the network card driver is not installed.  
`sudo lshw -c network`: see the model and vendor oationf the network card. For example, `BCM43142 802.11 b/g/m, vendor: Broadcom Corportion`  
For Broadcast network card:`sudo apt-get install`, `sudo apt-get bcmwl-kernel-source`
