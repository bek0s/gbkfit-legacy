#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "kernels_cuda_host.fatbin.c"
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll_25_kernels_cuda_host_cpp1_ii_664d8126(void) __attribute__((__constructor__));
static void __nv_cudaEntityRegisterCallback(void **__T22){__nv_dummy_param_ref(__T22);__nv_save_fatbinhandle_for_managed_rt(__T22);}
static void __sti____cudaRegisterAll_25_kernels_cuda_host_cpp1_ii_664d8126(void){__cudaRegisterBinary(__nv_cudaEntityRegisterCallback);}
