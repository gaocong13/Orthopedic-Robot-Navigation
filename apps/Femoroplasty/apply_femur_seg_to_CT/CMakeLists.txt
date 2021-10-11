
set(EXE_NAME "${XREG_EXE_PREFIX}apply-femur-seg-to-CT")

add_executable(${EXE_NAME} apply_femur_seg_to_CT_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
