file(REMOVE_RECURSE
  "../../../lib/libgbkfit_cuda_shared_d.pdb"
  "../../../lib/libgbkfit_cuda_shared_d.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/target_gbkfit_cuda_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
