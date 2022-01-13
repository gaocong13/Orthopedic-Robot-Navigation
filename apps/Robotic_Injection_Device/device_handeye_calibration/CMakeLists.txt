# in CMake we will have to set cisst_DIR to <cisst-build-tree>/cisst
find_package (cisst)

# make sure cisst is found
if (cisst_FOUND)

  # this will set the include and link directories
  # using all cisst settings
  include (${CISST_USE_FILE})

  set(EXE_NAME "${XREG_EXE_PREFIX}device-handeye-calibration")

  add_executable(${EXE_NAME} device_handeye_calibration_main.cpp)

  target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK} cisstCommon cisstVector)

else (cisst_FOUND)
  message (SEND_ERROR "Oops, cisst not found")
endif (cisst_FOUND) 

install(TARGETS ${EXE_NAME})

