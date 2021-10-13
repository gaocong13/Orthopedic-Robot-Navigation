
set(EXE_NAME "${XREG_EXE_PREFIX}femoroplasty-regi-singleview-pelvis-femur")

add_executable(${EXE_NAME} femoroplasty_regi_singleview_pelvis_and_femur_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
