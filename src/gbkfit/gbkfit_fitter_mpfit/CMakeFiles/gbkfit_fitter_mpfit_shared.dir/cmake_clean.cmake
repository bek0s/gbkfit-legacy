file(REMOVE_RECURSE
  "../../../lib/libgbkfit_fitter_mpfit_shared.pdb"
  "../../../lib/libgbkfit_fitter_mpfit_shared.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang C CXX)
  include(CMakeFiles/gbkfit_fitter_mpfit_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
