
set(EXE_NAME "${XREG_EXE_PREFIX}spine-device-multiconfig-optimhandeye")

add_executable(${EXE_NAME} device_multiconfig_optimhandeye_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
