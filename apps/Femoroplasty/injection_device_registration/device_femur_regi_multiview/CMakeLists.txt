
set(EXE_NAME "${XREG_EXE_PREFIX}device-femur-regi-multiview")

add_executable(${EXE_NAME} device_femur_regi_multiview_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
