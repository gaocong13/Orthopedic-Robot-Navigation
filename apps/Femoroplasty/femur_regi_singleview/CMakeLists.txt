
set(EXE_NAME "${XREG_EXE_PREFIX}femur-regi-singleview")

add_executable(${EXE_NAME} femur_regi_singleview_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
