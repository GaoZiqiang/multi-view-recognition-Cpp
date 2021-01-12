cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(MultiViewRecogCpp)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(MultiViewRecogCpp main.cpp DataManager.h)

find_package( OpenCV REQUIRED )#找到opencv源
target_link_libraries( MultiViewRecogCpp ${OpenCV_LIBS} )#添加opencv链接选项

target_link_libraries(MultiViewRecogCpp "${TORCH_LIBRARIES}")
set_property(TARGET MultiViewRecogCpp PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET MultiViewRecogCpp
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:MultiViewRecogCpp>)
endif (MSVC)
