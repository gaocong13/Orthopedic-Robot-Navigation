
set(EXE_NAME "${XREG_EXE_PREFIX}device-InjectorPolaris-handeye-reproj-bb")

add_executable(${EXE_NAME} device_InjectorPolaris_handeye_reproj_bb_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
