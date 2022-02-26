
set(EXE_NAME "${XREG_EXE_PREFIX}autocarm-proc-xraydcm-to-tiff")

add_executable(${EXE_NAME} autocarm_proc_xraydcm_to_tiff_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
