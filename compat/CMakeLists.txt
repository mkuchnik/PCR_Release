cmake_minimum_required(VERSION 3.1)

project(PCRs VERSION 1.0
        DESCRIPTION "Progressive Compressed Records"
        LANGUAGES C CXX)

add_compile_options(-W -Wall)

SET(Python_ADDITIONAL_VERSIONS 3 3.6 3.5 3.4 3.3 3.2 3.1 3.0)
find_package(PythonLibs 3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYTHON_INCLUDE_PATH "$ENV{HOME}/anaconda/envs/py36env/include/python3.6")
set(PYTHON_LIBRARIES "$ENV{HOME}/anaconda/envs/py36env/lib/libpython3.6m.so")

INCLUDE(FindProtobuf)
FIND_PACKAGE(Protobuf REQUIRED)
SET(Protobuf_USE_STATIC_LIBS ON)

ADD_SUBDIRECTORY(proto)

message("proto libs" ${PROTOBUF_LIBRARIES})
message("proto dirs" ${PROTOBUF_INCLUDE_DIRS})

add_library(pcr_header INTERFACE)
target_include_directories(pcr_header INTERFACE include)

add_library(eigen_3 INTERFACE)
target_include_directories(eigen_3 INTERFACE "${CMAKE_SOURCE_DIR}/eigen-git-mirror")

# Preselected variants of Greedy function templates
add_library(pcr STATIC src/pcr.cpp)
target_include_directories(pcr PUBLIC
  include
  proto
  ${PROTOBUF_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_include_directories(pcr PRIVATE src)
target_link_libraries(pcr PUBLIC
  eigen_3
  ${PROTOBUF_LIBRARIES}
)
target_compile_features(pcr PUBLIC cxx_std_11)

add_executable(main src/main.cpp)
target_include_directories(main PUBLIC
  include
  proto
  ${PROTOBUF_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(main PUBLIC
  pcr
  proto
  ${PROTOBUF_LIBRARIES}
)
target_compile_features(main PUBLIC cxx_std_11)

add_executable(jsk src/jsk.c)
set_property(TARGET jsk PROPERTY C_STANDARD 99)

add_executable(scan_only_jsk src/scan_only_jsk.c)
set_property(TARGET scan_only_jsk PROPERTY C_STANDARD 99)

add_subdirectory(pybind11)
pybind11_add_module(python_bindings python/python_bindings.cpp src/pcr.cpp)
target_include_directories(python_bindings PUBLIC
  include
  proto
  ${PROTOBUF_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_include_directories(python_bindings PRIVATE 
  src
)
target_link_libraries(python_bindings PUBLIC
  eigen_3
  proto
  ${PROTOBUF_LIBRARIES}
)
target_compile_features(python_bindings PUBLIC cxx_std_11)
# set_target_properties(python_bindings PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
