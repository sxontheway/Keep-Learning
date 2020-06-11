README
========
>Github .md文件自动目录生成： https://github.com/ekalinin/github-markdown-toc  

本文件仅展示md文件中`高频使用的`部分语法。  
本文件在Raw模式中的层级命名方法为 `一、 二、` --> `1. 2.` -->   `(1) （2）`。
*******
<br><br>

一、标题
=======
1.大标题，在下面用======（=个数无限制）
=======

2.中标题，在下面用------（-个数无限制）
-------

3.***、___、--- 可显示分段用的粗横线(个数三个及以上)，例如:
******
______
------

4.各种小标题(注意#后面要有一个空格)
# 一级标题  
## 二级标题  
### 三级标题  
#### 四级标题  
##### 五级标题  
###### 六级标题
*******
<br><br>




二、文本
=======
1.换行    
直接回车不能换行，需要上一行文本后面补两个空格，这样下一行的文本就换行了。  
或者在两行文本直接用\<br>加一个空行，但这个行间距有点大（其中\是转义字符，所以在Preview模式中不会显示出“\”）。<br>

2.字体  
斜体用`*abc*，_abc_`，例如 *abc*， _abc_  
粗体用`**abc**，__abc__`，例如 **粗体1**， __粗体2__  
删除线用`~~删除线~~`，例如 ~~删除线~~_  
斜体、粗体、删除线可混合使用<br>

3.文本块  
(1)方法一
	
	在一行开头加入1个Tab或者4个空格，得到高亮的单行。(上面没有标题时(###等标题指示符自带换行)，需要先回车再Tab)
	输入多行，每一行开头都加入1个Tab或者4个空格，得到高亮的文本块。
(2)方法二  
用一对各三个的反引号：  
```
一对单个的反引号，例如`abc`，会将高亮的内容显示在一行内。三对反引号将高亮多行。
文本块中的内容换行，直接回车即可，不再需要两个空格或<br>。
```   
<br>

4.代码块  
在三个反引号后面加上编程语言的名字，另起一行开始写代码，最后一行再加上三个反引号。例如：  
```Java
public static void main(String[]args){} //Java
```
```C
int main(int argc, char *argv[]) //C
```
```Bash
echo "hello GitHub" #Bash
```
```C++
string &operator+(const string& A,const string& B) //C++
```
<br>

5.块引用中的多级结构  
用`>`来表示层级结构，在表示文件路径时很常用。其中，换行用两个空格和一个回车。两个回车(也就是一行留白)跳出块引用(也可用于从低级目录返回高级目录)。例如：  
>/catkin_ws  
>>src  
>>>package1  
>>>>src  
>>>>CMakeLists.txt  
>>>>package.xml  
>>>pacakage2  
>>>>...

>>build  
>>devel  
<br>

6.列表  
(1)复选框列表  
`- [x]和- [ ]` 表示选择框。在md文件的Raw格式中，每个复选框要单独占一行，例如：
- [x] 这是一个md文件
- [ ] 不是一个md文件  

(2)单级有序列表  
数字+英文句点+一个空格：    
1. 项目一
2. 项目二
3. 项目三  
在列表内，文本内容的缩进将会与其所在的列表级别的缩进相同。  
当然，列表内的文本仍然用两个空格加回车换行。  

(3)多级有序列表，用一次Tab下降一级：  
1. 第一级
	1. 第二级
		1. 第三级  
		 第三级的第一行  
		 第三级的第二行  
		 要跳出列表回到普通文本，需要两个回车（在Raw文件中显式地有一行为空）
		 
(4)无序列表  
将有序列表的`数字+空格+英文句点`变为`*或-`即可，略。
*******
<br><br>




三、图片链接及表格
=======
1. 图片（**有感叹号**）
	>基本格式(其中alt和title都可以省略)：  
	>>`![Alt](URL Title)`

	其中，`Title`表示鼠标悬停在图片时的显示文本(注意这里要加引号)。  
	`URL`为图片地址，例如http://www.baidu.com/img/bdlogo.gif 。如果是本repository内的内容，可使用相对路径。  
	`Alt`为图片加载失败时候，替换的文本。例如：  
	输入`![baidu](http://www.baidu.com/img/bdlogo.gif "百度logo") `得到：  
	![baidu](http://www.baidu.com/img/bdlogo.gif "百度logo")

1. 另一种链接图片的方式  
	一张图
	```
	<p align="center" >
		<img src="http://www.baidu.com/img/bdlogo.gif" width="200" height="100">
	</p>
	```

	<p align="center" >
		<img src="http://www.baidu.com/img/bdlogo.gif" width="200" height="100">
	</p>

	或多张图并排：
	```
	<center class="half">
		<img  src="http://www.baidu.com/img/bdlogo.gif" width=300>
		<img  src="http://www.baidu.com/img/bdlogo.gif" width=300>
	</center>
	```
	<center class="half">
		<img  src="http://www.baidu.com/img/bdlogo.gif" width=300>
		<img  src="http://www.baidu.com/img/bdlogo.gif" width=300>
	</center>


1. 链接（**没有感叹号**）  
	>`[Alt](URL Title)`

	此时`URL`链接到一个网址。`Alt`为显示的东西，不一定是一个文本，也可以是一个图片。例如输入：  
	`[![Udacity](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)`  
	得到：  
	[![Udacity](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)<br><br>

1. 另一种链接方式。  
	例如，先在文末定义 `[zhihu]:https://www.zhihu.com "知乎官网"`，再输入`[Github][Github]`，得到：

	>[Github][Github]
	<br>

1. 锚点(可用于页内跳转，构成目录）
	```
	基本语法：[Alt](#Tiltle)
	其中Alt为要显示的东西， Title为本文件的某个标题
	```
	Title中的英文字母都被转化为小写字母了，例如输入：
	```
	[README](#readme)<br>
	[一级标题](#一级标题)
	```
	可得到：  
	[README](#readme)<br>
	[一级标题](#一级标题)
	>具体的Title该怎么写，可以将鼠标移到标题上（有#，##，###等前缀的是标题），参照屏幕下方出现的URL即可。
	<br>

1. 表格  
	基本格式(表格开始和结束之前都要有回车):

	| 表头1  | 表头2|
	| ---------- | -----------|
	| 表格单元   | 表格单元   |
	| 表格单元   | 表格单元   |

	对齐：

	| 左对齐 | 居中  | 右对齐 |
	| :------------ |:---------------:| -----:|
	| col 3 is      | some wordy text | $1600 |
	| col 2 is      | centered        |   $12 |
	| zebra stripes | are neat        |    $1 |

	<br>

1. diff语法  
	其语法与代码块类似，只是在三个反引号后面写diff：
	```diff
	+ 鸟宿池边树
	- 僧推月下门
	```
	----------
	
	[Github]:https://www.github.com "我的github"
