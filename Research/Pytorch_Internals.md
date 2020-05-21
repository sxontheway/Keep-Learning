# Pytorch Internals
> https://speakerdeck.com/perone/pytorch-under-the-hood?slide=21

## Tensors
* Although PyTorch has an elegant python first design, all PyTorch heavy work is actually implemented in C++. The integration of C++ code is usually done using what is called an **extension**.

* zero-copy tensors
    ```python
    # a copy is made
    np_array = np.ones((1,2))
    torch_array = torch.tensor(np_array)    # This make a copy
    torch_array.add_(1.0)   # underline after an operation means an in-place operation
    print(np_array)     # array([[1., 1.]])

    # zero-copy
    np_array = np.ones((1,2))
    torch_array = torch.from_numpy(np_array)    # This make a copy
    torch_array.add_(1.0) # or torch_array += 1.0 (in place operation)
    print(np_array)     # array([[2., 2.]])

    # zero-copy
    np_array = np.ones((1,2))
    torch_array = torch.from_numpy(np_array)    # This make a copy
    torch_array = torch_array + 1.0     # not an in-place operatio on torch_array 
    print(np_array)     # array([[1., 1.]])
    ```
    The tensor **FloatTensor** did a copy of the **numpy array data pointer** instead of the contents. The reference is kept safe by Python reference counting mechanism.

* The abstraction responsible for holding the data isn't actually the Tensor, but the Storage. We can have multiple tensors sharing the same storage, but with different interpretations, also called views, but without duplicating memory.

    ```python 
    t_a = torch.ones((2,2))
    t_b = t_a.view(4)
    t_a_data = t_a.storage().data_ptr()
    t_b_data = t_b.storage().data_ptr()
    t_a_data == t_b_data

    # True
    ```

## JIT: just-in-time compiler 