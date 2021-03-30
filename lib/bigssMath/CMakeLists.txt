file(GLOB_RECURSE XREG_CUR_LIB_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(xreg_bigssMath OBJECT ${XREG_CUR_LIB_HEADERS}
                                   bigssMath.cpp
				   bigssMathEigen.cpp
				   interpolatingSpline.cpp
				   smoothingSpline.cpp
				   slicerUtil.cpp
				   bigssKinematicsUtils.cpp
				   bigss_debug_util.cpp)
