file(REMOVE_RECURSE
  "../../../lib/libgbkfit_static.pdb"
  "../../../lib/libgbkfit_static.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/gbkfit_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
