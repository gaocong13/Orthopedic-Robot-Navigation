
set(EXE_NAME "${XREG_EXE_PREFIX}osteonecrosis-example-drr-regi")

add_executable(${EXE_NAME} example_drr_regi_singleview_femur_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
