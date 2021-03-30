
project(jhmr_spine_common)

add_library(jhmr-spine-common CommonFEM.cpp)

target_link_libraries(jhmr-spine-common jhmr_static)

JHMR_ADD_ALL_DEPENDENCIES(jhmr-spine-common)

