file(REMOVE_RECURSE
  "../../../lib/libgbkfit_cuda_shared.pdb"
  "../../../lib/libgbkfit_cuda_shared.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/gbkfit_cuda_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
