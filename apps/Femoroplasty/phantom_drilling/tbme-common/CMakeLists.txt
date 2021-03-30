
project(jhmr_tbme_common)

add_library(jhmr-tbme-common CommonTBME.cpp)

target_link_libraries(jhmr-tbme-common jhmr_static)

JHMR_ADD_ALL_DEPENDENCIES(jhmr-tbme-common)

