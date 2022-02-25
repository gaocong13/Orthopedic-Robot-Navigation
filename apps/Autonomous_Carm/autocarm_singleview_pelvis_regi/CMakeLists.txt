
set(EXE_NAME "${XREG_EXE_PREFIX}autocarm-singleview-pelvis-regi")

add_executable(${EXE_NAME} autocarm_singleview_pelvis_regi_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
