
set(EXE_NAME "${XREG_EXE_PREFIX}spine-spine-sim-regi-multiview-vertb")

add_executable(${EXE_NAME} spine_sim_regi_multiview_vertb_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
