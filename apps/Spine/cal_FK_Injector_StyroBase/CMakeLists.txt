
set(EXE_NAME "${XREG_EXE_PREFIX}cal-FK-Injector-StyroBase")

add_executable(${EXE_NAME} cal_FK_Injector_StyroBase_main.cpp)

target_link_libraries(${EXE_NAME} PUBLIC ${XREG_EXE_LIBS_TO_LINK})

install(TARGETS ${EXE_NAME})
