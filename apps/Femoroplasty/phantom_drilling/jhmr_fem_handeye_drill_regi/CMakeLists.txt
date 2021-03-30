
cmake_minimum_required(VERSION 2.8)

project(jhmr_fem_handeye_drill_regi)

set(exe_name jhmr-fem-handeye-drill-regi)

if (NOT JHMR_TOP_LEVEL_BUILD)
  set(JHMR_USE_ITK ON)
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/jhmr_setup.cmake)
endif (NOT JHMR_TOP_LEVEL_BUILD)

add_executable(${exe_name} ${CMAKE_CURRENT_SOURCE_DIR}/jhmr_fem_handeye_drill_regi_main.cpp)

JHMR_ADD_ALL_DEPENDENCIES(${exe_name})

target_link_libraries(${exe_name} ${JHMR_LIBS_TO_LINK})

install(TARGETS ${exe_name} RUNTIME DESTINATION bin)
