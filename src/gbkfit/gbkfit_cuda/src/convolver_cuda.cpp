
#include "gbkfit/cuda/convolver_cuda.hpp"
#include "gbkfit/cuda/ndarray_cuda.hpp"


namespace gbkfit {
namespace cuda {


convolver_cuda::convolver_cuda(const ndshape& shape, const ndarray* psf, const ndarray* lsf)
{
    ndarray_cuda* m_data_psf = new ndarray_cuda_device(psf->get_shape());
    ndarray_cuda* m_data_lsf = new ndarray_cuda_device(lsf->get_shape());

    m_data_psf->copy_data(psf);
    m_data_lsf->copy_data(lsf);


    ndshape m_shape({shape[0]+psf->get_shape()[0]-1,
                     shape[1]+psf->get_shape()[1]-1,
                     shape[2]+psf->get_shape()[2]-1});


}

void convolver_cuda::convolve(ndarray* data)
{

}

} // namespace cuda
} // namespace gbkfit
