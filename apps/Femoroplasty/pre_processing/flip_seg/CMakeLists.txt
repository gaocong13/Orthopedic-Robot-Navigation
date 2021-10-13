
set(EXE_NAME "${XREG_EXE_PREFIX}flip-seg")

add_executable(${EXE_NAME} flip_seg_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
