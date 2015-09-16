#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "kernels_cuda_device.fatbin.c"
extern void __device_stub__ZN6gbkfit6models9galaxy_2d19kernels_cuda_device3fooEPfS3_ii(float *, float *, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll_27_kernels_cuda_device_cpp1_ii_6d53e175(void) __attribute__((__constructor__));
void __device_stub__ZN6gbkfit6models9galaxy_2d19kernels_cuda_device3fooEPfS3_ii(float *__par0, float *__par1, int __par2, int __par3){__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 20UL);__cudaLaunch(((char *)((void ( *)(float *, float *, int, int))gbkfit::models::galaxy_2d::kernels_cuda_device::foo)));}
# 10 "/home/bekos/code/gbkfit/gbkfit/src/gbkfit/gbkfit_models_galaxy_2d_cuda/src/kernels_cuda_device.cu"
void gbkfit::models::galaxy_2d::kernels_cuda_device::foo( float *__cuda_0,float *__cuda_1,int __cuda_2,int __cuda_3)
# 14 "/home/bekos/code/gbkfit/gbkfit/src/gbkfit/gbkfit_models_galaxy_2d_cuda/src/kernels_cuda_device.cu"
{__device_stub__ZN6gbkfit6models9galaxy_2d19kernels_cuda_device3fooEPfS3_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3);

}
# 1 "kernels_cuda_device.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T20) {  __nv_dummy_param_ref(__T20); __nv_save_fatbinhandle_for_managed_rt(__T20); __cudaRegisterEntry(__T20, ((void ( *)(float *, float *, int, int))gbkfit::models::galaxy_2d::kernels_cuda_device::foo), _ZN6gbkfit6models9galaxy_2d19kernels_cuda_device3fooEPfS3_ii, (-1)); }
static void __sti____cudaRegisterAll_27_kernels_cuda_device_cpp1_ii_6d53e175(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }
