file(REMOVE_RECURSE
  "../../../lib/libgbc_static_d.pdb"
  "../../../lib/libgbc_static_d.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/target_gbc_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
