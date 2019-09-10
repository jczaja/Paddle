# Design Doc: MKL-DNN Operators

PaddlePaddle Operators with  MKL-DNN kernels are using highly optimized Intel MKL-DNN library to compute PaddlePaddle operators functionality. MKL-DNN API does
provide *primitives* that are implementing operators functionality. Not every type of PaddlePaddle operator has its equivalent in MKL-DNN so PaddePaddle MKL-DNN kernels
are provided only to subset of all PaddlePaddle operators. 

To extend operator with MKL-DNN kernel three modifications are required:
* MKL-DNN Kernel 
* API Change 
* Support for loading MKL-DNN kernels in *framework::OpKernelType GetExpectedKernelType()* 

### MKL-DNN kernel

To add MKL-DNN kernel to operator Classes that inherit after OpKernel<T> has to be created (one for forward op and one for backward op) that contain *Compute* methods.
Example of Softmax MKL-DNN kernel declaration :

`
template <typename T>
class SoftmaxMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
`




Once MKL-DNN kernel is added it has to be registered so Paddle is aware of its existance. Example of registration for softmax mkl-dnn kernels
`
REGISTER_OP_KERNEL(softmax, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNKernel<float>);
REGISTER_OP_KERNEL(softmax_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNGradKernel<float>);
`

### API Change 
To execute operator MKL-DNN kernel operator needs to have attribute *use_mkldnn* and when it its value is set to _true_ only then MKL-DNN kernel is to be executed.



### Support for loading MKL-DNN kernels in *framework::OpKernelType GetExpectedKernelType()* 
`
#ifdef PADDLE_WITH_MKLDNN
    if (library_ == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library_ = framework::LibraryType::kMKLDNN;
      layout_ = framework::DataLayout::kMKLDNN;
    }
#endif
`
