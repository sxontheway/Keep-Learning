目录：https://github.com/BBuf/tvm_mlir_learn

<br> 
<br>

# tvm.relay.frontend 
> https://mp.weixin.qq.com/s/KFxd3zf76EP3DFcCAPZjvQ

从 ONNX model 如何转成 Relay 表达式：

* [如图所示：](./relay_frontend.svg) 从 `from_onnx()` 一直可以追到 `_make.conv2d()`
* 从 `_make.conv2d()` 进一步追踪：`/tvm/python/tvm/relay/op/nn/_make.py` 中 `tvm._ffi._init_api("relay.op.nn._make", __name__)`
* `tvm._ffi._init_api()` 的定义在 `tvm/python/tvm/_ffi/registry.py`中。里面调用了 `get_global_func()`，会加载编译好的 TVM 动态库获取这个动态库里面的函数名称来进行匹配。获取到 C++ 注册的相应函数后就可以设置到 `_make.py` 文件中，即相当于在 `_make.py` 中定义了 conv2d 算子的函数了
    * 反推回去，`from_onnx()` 就拿到了一个 `tvm.IRModule`，其中有来源于 `_make.py` 的函数
    * conv2d 算子的注册代码在 `tvm/src/relay/op/nn/convolution.cc`；在一个 op 类中实际上并没有包含这个 op 的计算过程，op 中的一系列操作只是拿到了 Relay 卷积 op 的 IR，和的输入输出以及属性的信息；op 的计算过程是在 TVM 的 TOPI 中完成的

<br>
<br>

# TOPI
> > https://mp.weixin.qq.com/s/1YlTSUArDIzY-9zeUAIfhQ
* [TVM Codebase Walkthrough by Example](https://tvm.apache.org/docs/dev/tutorial/codebase_walkthrough.html)

    `src/relay` is the component that manages a computational graph. Nodes in a graph are compiled and executed using infrastructure implemented in the rest of `src`. python provides python bindings for the C++ API and driver code that users can use to execute compilation. Operators corresponding to each node are registered in `src/relay/op`. Implementations of operators are in `topi`, and they are **coded in either C++ or Python**.  
    
    简而言之，TVM 中的算子实现都在 topi 中，其中有一些是用 python 实现的

* 例子：`tvm/python/tvm/relay/op/image/` 文件夹
    * `tvm/python/tvm/relay/op/image/image.py`：文件名前无 `_`，是用于 frontend 从 python 调 C++ querying operator registry 中的一步，见流程图。例如，文件中的一个函数 `op.nn.conv2d()` ，进一步调用 `_make.conv2d()`

    * `tvm/python/tvm/relay/op/image/_image.py`：文件名前有 `_`，主要建立了 OP 和 TOPI 算子的连接。例如在 `topi.image.resize1d()` 中有具体实现，其返回值是一个 `tvm.te.Tensor`

        ```python
        # resize
        @reg.register_compute("image.resize1d")
        def compute_resize1d(attrs, inputs, out_type):
            """compute definition for resize1d op"""
            size = attrs.size
            roi = attrs.roi
            layout = attrs.layout
            method = attrs.method
            coord_trans = attrs.coordinate_transformation_mode
            rounding_method = attrs.rounding_method
            cubic_alpha = attrs.cubic_alpha
            cubic_exclude = attrs.cubic_exclude
            extrapolation_value = attrs.extrapolation_value
            out_dtype = attrs.out_dtype
            return [
                topi.image.resize1d(
                    inputs[0],
                    roi,
                    size,
                    layout,
                    method,
                    coord_trans,
                    rounding_method,
                    cubic_alpha,
                    cubic_exclude,
                    extrapolation_value,
                    out_dtype,
                )
            ]

        reg.register_injective_schedule("image.resize1d")
        ```

        * `reg.register_injective_schedule("image.resize1d")` 完成了 schedule primitive。一些 scedule template 例子见 `tvm/python/tvm/topi/gpu/conv2d_nhwc.py`，`tvm/python/tvm/topi/arm_cpu/injective.py`

