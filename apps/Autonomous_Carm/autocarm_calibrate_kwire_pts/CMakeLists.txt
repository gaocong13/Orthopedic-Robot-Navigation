
set(EXE_NAME "${XREG_EXE_PREFIX}autocarm-calibrate-kwire-pts")

add_executable(${EXE_NAME} autocarm_calibrate_kwire_pts_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
