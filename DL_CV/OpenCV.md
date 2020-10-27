# 1. OpenCV头文件
* cv.hpp和opencv.hpp是等同的关系。只不过，前者是早期opencv版本中的定义名称，而后者opencv.hpp则是3.0版本之后的表示方法。
* cv.hpp和opencv.hpp都包含了一下头文件
  * #include <opencv2/core.hpp>
  * #include <opencv2/imgproc.hpp>
  * #include <opencv2/video.hpp>
  * #include <opencv2/objdetect.hpp>
  * #include <opencv2/imgcodecs.hpp>
  * #include <opencv2/highgui.hpp>
  * #include <opencv2/ml.hpp>
  
简言之，我们在编写代码的时候，或许只需要一个简单的`#include<opencv2/opencv.hpp>`就可以轻松的解决红色波浪线未定义字符的烦恼。
`.hpp`和`.h`文件差别是`.hpp`把还包含了函数实现。


------
<br><br>

# 2. Mat和Mat_
## 2.1 Mat
* 图像的表示：最老的C接口的OpenCV版本中，用 ***IplImage*** ，需要手动管理内存。C++接口的OpenCV中，用 ***Mat*** ，不用手动管理内存。
* 浅拷贝：Mat对象的赋值运算或拷贝构造函数只拷贝Header，不拷贝矩阵数据，对一个对象的修改会影响其他对象。
```C==
Mat A, C;                          // creates just the header parts
A = imread(argv[1], IMREAD_COLOR); // here we'll know the method used (allocate matrix)
Mat B(A);                          // Use the copy constructor
C = A;                             // Assignment operator
```
* 深拷贝：可用`cv::Mat::clone()`和`cv::Mat::copyTo()`实现图像矩阵数据的拷贝。
```C++
Mat F = A.clone();    
Mat G;
A.copyTo(G);
```
* 除此之外，cv::Mat还有如下常用属性： 
API reference: https://docs.opencv.org/3.1.0/d3/d63/classcv_1_1Mat.html

|成员函数|解释|
| :------------ | :-----|
|void cv::Mat::create (int rows, int cols, int type)|创建矩阵|
| _Tp& cv::Mat::at (int i0, int i1)|例如用`image.at<Vec3b>(y,x)[c]`访问y行x列c通道的值|
|uchar * ptr<T>()|行指针，例如`image.ptr<uchar>(1)`为指向第1行首地址的指针|
|int cv::Mat::channels () const|矩阵元素的通道数，例如RGB为三个通道|
|int cv::Mat::type () const|它是一系列的预定义的常量，命名规则为`CV_+（位数）+（数据类型）+ C +（通道数）`，例如`CV_8UC1`|
|int cv::Mat::depth () const|返回矩阵元素的深度，例如一个8-bit signed element array，返回`CV_8U`|
 
|成员变量|解释|
| :------------ | :-----|
|uchar* cv::Mat::data| uchar类型的指针，指向Mat数据矩阵的首地址|
|int cv::Mat::dims|Mat矩阵的维度，若Mat是一个二维矩阵，则 dims=2，三维则 dims=3|
|int cv::Mat::cols|矩阵列数，如果 dims>2 为-1|
|int cv::Mat::rows |矩阵行数，如果 dims>2 为-1|
|cv::Mat::**step**|步长，指为了便于访问矩阵元素，需要移动多少距离，例如多幅同样大小的图像组成的3维矩阵: <br> step[0] 指从当前图像开始处到下一图像的开始处跨越多少字节 <br> step[1] 指从当前图像行首到下一行首相距多少字节 <br> step[2] 从当前像素到下一像素步长是多少 <br> image.step这样的用法也就是image.step[0]的意思|
|cv::Mat:: **size**|是指多维矩阵中每一维的大小，例如多幅同样大小的图像组成的3维矩阵:  <br> size[0] 指有共有多少幅图像 <br> size[1] 指单幅图像有多少行 <br> size[3] 指单幅图像有多少列  | 

* `cv::Mat`下图像的遍历方法：
```C++
//*********遍历方法1：.at 操作***********//
for(int y = 0; y < image.rows; y++) {
    for( int x = 0; x < image.cols; x++) {
        for( int c = 0; c < image.channels(); c++) {
            new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(alpha*image.at<Vec3b>(y,x)[c] + beta);
            // saturate_cast<>实现从一个类型到另一个类型的映射（saturate的意思是按比例缩放至saturation）
        }
    }
}
 
//*********更高效的遍历方法：用行指针***************//
for (size_t y=0; y<image.rows; y++) {     // size_t在C语言中就有了，是一种用来记录大小的数据类型，此处也可用int代替 
    unsigned char* row_ptr = image.ptr<unsigned char> (y); // row_ptr是第y行的头指针
    for（size_t x=0; x<image.cols; x++）{
        unsigned char* data_ptr = &row_ptr[x * image.channels()]; // data_ptr为指向像素数据的指针
        for (int c = 0; c != image.channels(); c++) {
            unsigned char data = data_ptr[c]; // data为I(x,y)第c个通道的值
        }
    }
}


// unsigned char范围为0-255。 eg. 若image.depth() == CV_8U，则每个像素每个通道都是用0-255表示  
// OpenCV库中定义Vec3b的语句：`typedef Vec<uchar, 3> cv::Vec3b`，所以`Vec3b`表示一个uchar类型的数组，长度为3。
    //其中`cv::Vec< _Tp, cn >`是OpenCV中定义的一个类，继承了`cv::Matx< _Tp, m, n >`这个类。
```

## 2.2 Mat_
一些代码中用到cv::Mat_，Mat_继承了Mat类，但比Mat类并没有更多的东西。只是在编译前就能确定元素类型时，用Mat_会更方便。  
>例如：  
>Mat访问元素调用的是 Mat::at(int y,int x)， `Mat image(100,100,CV_8U);  image.at(1,2) = 2;`  
>Mat_访问元素调用的是 Mat_::operator()(int y,int x))， `Mat_<double> image(20,20);  image(1,2) = 3;`
```C++
template<typename T> class Mat_ : public Mat
{
public:
    // ... some specific methods
    //         and
    // no new extra fields
};
```

------
<br><br>

# 3. 实践：用OpenCV读取多个读取多个IP摄像头的视频流
> https://zhuanlan.zhihu.com/p/38136322  
