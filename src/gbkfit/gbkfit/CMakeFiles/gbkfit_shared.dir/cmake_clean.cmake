file(REMOVE_RECURSE
  "../../../lib/libgbkfit_shared.pdb"
  "../../../lib/libgbkfit_shared.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/gbkfit_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
