
set(EXE_NAME "${XREG_EXE_PREFIX}newsnake-sim-singleview")

add_executable(${EXE_NAME} newsnake_sim_singleview_main.cpp)

target_link_libraries(${EXE_NAME} ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
