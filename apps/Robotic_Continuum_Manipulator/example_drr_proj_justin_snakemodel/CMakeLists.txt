
set(EXE_NAME "${XREG_EXE_PREFIX}snake-example-drr-proj-justin-snakemodel")

add_executable(${EXE_NAME} snake_example_drr_proj_justin_snakemodel_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
