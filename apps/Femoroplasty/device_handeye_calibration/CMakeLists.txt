
set(EXE_NAME "${XREG_EXE_PREFIX}device-handeye-calibration")

add_executable(${EXE_NAME} device_handeye_calibration_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
