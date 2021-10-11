
set(EXE_NAME "${XREG_EXE_PREFIX}spine-device-sim-multiconfig-multiview")

add_executable(${EXE_NAME} device_sim_multiconfig_multiview_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
