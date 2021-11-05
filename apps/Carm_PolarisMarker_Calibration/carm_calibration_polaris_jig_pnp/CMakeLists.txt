# in CMake we will have to set cisst_DIR to <cisst-build-tree>/cisst
find_package (cisst)

# make sure cisst is found
if (cisst_FOUND)
  # this will set the include and link directories
  # using all cisst settings
  set(EXE_NAME "${XREG_EXE_PREFIX}carm-calibration-polaris-jig-pnp")

  add_executable(${EXE_NAME} carm_calibration_polaris_jig_pnp_main.cpp)

  target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

  install(TARGETS ${EXE_NAME})

else (cisst_FOUND)
  message (SEND_ERROR "Oops, cisst not found")
endif (cisst_FOUND) 




