
set(EXE_NAME "${XREG_EXE_PREFIX}device-bb-pnp")

add_executable(${EXE_NAME} device_bb_pnp_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
