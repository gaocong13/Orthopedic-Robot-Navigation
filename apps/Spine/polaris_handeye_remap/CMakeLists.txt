
set(EXE_NAME "${XREG_EXE_PREFIX}spine-polaris-handeye-remap")

add_executable(${EXE_NAME} polaris_handeye_remap_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
