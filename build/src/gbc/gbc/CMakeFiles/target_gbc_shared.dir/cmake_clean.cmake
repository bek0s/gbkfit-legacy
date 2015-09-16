file(REMOVE_RECURSE
  "../../../lib/libgbc_shared_d.pdb"
  "../../../lib/libgbc_shared_d.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/target_gbc_shared.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
