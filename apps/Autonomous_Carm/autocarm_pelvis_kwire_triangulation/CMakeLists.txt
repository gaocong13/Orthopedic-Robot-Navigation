
set(EXE_NAME "${XREG_EXE_PREFIX}autocarm-pelvis-kwire-triangulation")

add_executable(${EXE_NAME} autocarm_pelvis_kwire_triangulation_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
