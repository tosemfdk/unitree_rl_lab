cmake_minimum_required(VERSION 3.12)
project(go2_controller)

set(UNITREE_SDK2_ROOT "")
if(DEFINED ENV{UNITREE_SDK2_DIR} AND EXISTS "$ENV{UNITREE_SDK2_DIR}")
  set(UNITREE_SDK2_ROOT "$ENV{UNITREE_SDK2_DIR}")
elseif(EXISTS "${PROJECT_SOURCE_DIR}/../../../../unitree_sdk2")
  set(UNITREE_SDK2_ROOT "${PROJECT_SOURCE_DIR}/../../../../unitree_sdk2")
endif()

set(UNITREE_SDK2_INCLUDE_DIR "/usr/local/include")
if(UNITREE_SDK2_ROOT AND EXISTS "${UNITREE_SDK2_ROOT}/include/unitree/dds_wrapper/robots/go2/go2.h")
  set(UNITREE_SDK2_INCLUDE_DIR "${UNITREE_SDK2_ROOT}/include")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(UNITREE_ARCH_DIR "aarch64")
else()
  set(UNITREE_ARCH_DIR "x86_64")
endif()

find_library(UNITREE_SDK2_LIB NAMES unitree_sdk2 PATHS /usr/local/lib /usr/lib NO_DEFAULT_PATH)
find_library(DDSC_LIB NAMES ddsc PATHS /usr/local/lib /usr/lib NO_DEFAULT_PATH)
find_library(DDSCXX_LIB NAMES ddscxx PATHS /usr/local/lib /usr/lib NO_DEFAULT_PATH)

if(NOT UNITREE_SDK2_LIB AND UNITREE_SDK2_ROOT)
  if(EXISTS "${UNITREE_SDK2_ROOT}/lib/${UNITREE_ARCH_DIR}/libunitree_sdk2.a")
    set(UNITREE_SDK2_LIB "${UNITREE_SDK2_ROOT}/lib/${UNITREE_ARCH_DIR}/libunitree_sdk2.a")
  endif()
endif()
if(NOT DDSC_LIB AND UNITREE_SDK2_ROOT)
  if(EXISTS "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH_DIR}/libddsc.so")
    set(DDSC_LIB "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH_DIR}/libddsc.so")
  endif()
endif()
if(NOT DDSCXX_LIB AND UNITREE_SDK2_ROOT)
  if(EXISTS "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH_DIR}/libddscxx.so")
    set(DDSCXX_LIB "${UNITREE_SDK2_ROOT}/thirdparty/lib/${UNITREE_ARCH_DIR}/libddscxx.so")
  endif()
endif()

if(NOT UNITREE_SDK2_LIB)
  message(FATAL_ERROR "unitree_sdk2 library not found. Set UNITREE_SDK2_DIR or install to /usr/local/lib.")
endif()
if(NOT DDSC_LIB OR NOT DDSCXX_LIB)
  message(FATAL_ERROR "CycloneDDS libraries not found (ddsc/ddscxx). Set UNITREE_SDK2_DIR or install to /usr/local/lib.")
endif()

set(ONNXRUNTIME_LIB_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib")
find_library(ONNXRUNTIME_LIB NAMES onnxruntime PATHS ${ONNXRUNTIME_LIB_DIR} NO_DEFAULT_PATH)
if(NOT ONNXRUNTIME_LIB)
  if(EXISTS "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1")
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1")
  elseif(EXISTS "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1.22.0")
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1.22.0")
  else()
    message(FATAL_ERROR "onnxruntime library not found in ${ONNXRUNTIME_LIB_DIR}")
  endif()
endif()

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

include_directories(
  /usr/include/eigen3
  ${UNITREE_SDK2_INCLUDE_DIR}
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  ${UNITREE_SDK2_ROOT}/thirdparty/include
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  ${UNITREE_SDK2_LIB} ${DDSC_LIB} ${DDSCXX_LIB} rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${ONNXRUNTIME_LIB}
)

file(GLOB_RECURSE ADD_SRC_LIST
  src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_ctrl main.cpp)
