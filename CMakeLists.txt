cmake_minimum_required(VERSION 3.22)

project(RavePerformanceInstrument VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RAVE_USE_SYSTEM_JUCE "Use a JUCE CMake package already installed on the system" ON)
option(RAVE_ENABLE_LIBTORCH "Build the TorchScript model backend when LibTorch is available" ON)

if(RAVE_USE_SYSTEM_JUCE)
    find_package(JUCE CONFIG REQUIRED)
else()
    include(FetchContent)
    FetchContent_Declare(
        JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 7.0.9
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(JUCE)
endif()

add_library(rave_engine STATIC
    src/engine/AudioBlockQueue.cpp
    src/engine/AudioBlockQueue.h
    src/engine/InferenceWorker.cpp
    src/engine/InferenceWorker.h
    src/engine/RaveAudioEngine.cpp
    src/engine/RaveAudioEngine.h
    src/model/BackgroundModelLoader.cpp
    src/model/BackgroundModelLoader.h
    src/model/ModelBackend.h
)

target_include_directories(rave_engine PUBLIC src)

target_link_libraries(rave_engine
    PUBLIC
        juce::juce_audio_basics
        juce::juce_audio_devices
    PRIVATE
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
)

if(RAVE_ENABLE_LIBTORCH)
    find_package(Torch QUIET)
    if(Torch_FOUND)
        target_sources(rave_engine PRIVATE
            src/model/TorchScriptBackend.cpp
            src/model/TorchScriptBackend.h
        )
        target_link_libraries(rave_engine PUBLIC "${TORCH_LIBRARIES}")
        target_compile_definitions(rave_engine PUBLIC RAVE_HAS_LIBTORCH=1)
    else()
        message(STATUS "LibTorch not found; TorchScript backend disabled")
    endif()
endif()

target_compile_definitions(rave_engine
    PUBLIC
        JUCE_USE_CURL=0
        JUCE_WEB_BROWSER=0
)

include(CTest)
if(BUILD_TESTING)
    add_executable(rave_engine_tests tests/AudioBlockQueueTests.cpp)
    target_link_libraries(rave_engine_tests PRIVATE rave_engine)
    add_test(NAME rave_engine_tests COMMAND rave_engine_tests)

    add_executable(rave_inference_worker_tests tests/InferenceWorkerTests.cpp)
    target_link_libraries(rave_inference_worker_tests PRIVATE rave_engine)
    add_test(NAME rave_inference_worker_tests COMMAND rave_inference_worker_tests)

    add_executable(rave_audio_engine_tests tests/RaveAudioEngineTests.cpp)
    target_link_libraries(rave_audio_engine_tests PRIVATE rave_engine)
    add_test(NAME rave_audio_engine_tests COMMAND rave_audio_engine_tests)

    add_executable(rave_model_loader_tests tests/BackgroundModelLoaderTests.cpp)
    target_link_libraries(rave_model_loader_tests PRIVATE rave_engine)
    add_test(NAME rave_model_loader_tests COMMAND rave_model_loader_tests)

    if(Torch_FOUND)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_executable(rave_torch_backend_tests tests/TorchScriptBackendTests.cpp)
            target_link_libraries(rave_torch_backend_tests PRIVATE rave_engine)
            add_test(
                NAME rave_torch_backend_tests
                COMMAND "${Python3_EXECUTABLE}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_torch_backend_test.py"
                        "$<TARGET_FILE:rave_torch_backend_tests>"
            )
        endif()
    endif()
endif()

juce_add_gui_app(rave_instrument
    PRODUCT_NAME "RAVE Performance Instrument"
    COMPANY_NAME "RAVE Performance Instrument"
    BUNDLE_ID "org.raveperformance.instrument"
)

juce_generate_juce_header(rave_instrument)

target_sources(rave_instrument PRIVATE src/app/Main.cpp)

target_link_libraries(rave_instrument
    PRIVATE
        rave_engine
        juce::juce_gui_extra
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
