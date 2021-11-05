set(EXE_NAME "${XREG_EXE_PREFIX}carm-calibration-proj-data")

add_executable(${EXE_NAME} xreg_carm_calibration_proj_data_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
