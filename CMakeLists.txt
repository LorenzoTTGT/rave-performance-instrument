cmake_minimum_required(VERSION 3.22)

project(RavePerformanceInstrument VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RAVE_USE_SYSTEM_JUCE "Use a JUCE CMake package already installed on the system" ON)
option(RAVE_ENABLE_LIBTORCH "Build the TorchScript model backend when LibTorch is available" ON)

if(RAVE_USE_SYSTEM_JUCE AND NOT CMAKE_DISABLE_FIND_PACKAGE_JUCE)
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
    src/engine/LifecycleStatusText.h
    src/engine/RaveAudioEngine.cpp
    src/engine/RaveAudioEngine.h
    src/app/StandalonePresetModelCoordinator.cpp
    src/app/StandalonePresetModelCoordinator.h
    src/model/BackgroundModelLoader.cpp
    src/model/BackgroundModelLoader.h
    src/model/ModelBackend.h
    src/model/ModelQualification.cpp
    src/model/ModelQualification.h
    src/state/StandaloneSessionState.cpp
    src/state/StandaloneSessionState.h
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

set(RAVE_ICON_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/assets/icon/rave-instrument-icon.svg")
set(RAVE_ICON_SMALL "${CMAKE_CURRENT_SOURCE_DIR}/assets/icon/rave-instrument-icon-32.png")
set(RAVE_ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/assets/icon/rave-instrument-icon-1024.png")

# The SVG is the canonical project-owned artwork used by both editor targets.
# JUCE's inspected ICON_BIG/ICON_SMALL facilities below create platform icons.
juce_add_binary_data(rave_icon_resources
    NAMESPACE RaveIconAssets
    HEADER_NAME RaveIconAssets.h
    SOURCES "${RAVE_ICON_SOURCE}"
)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME rave_icon_assets_tests
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/IconAssetTests.py")
    add_executable(rave_engine_tests tests/AudioBlockQueueTests.cpp)
    target_link_libraries(rave_engine_tests PRIVATE rave_engine)
    add_test(NAME rave_engine_tests COMMAND rave_engine_tests)

    add_executable(rave_inference_worker_tests tests/InferenceWorkerTests.cpp)
    target_link_libraries(rave_inference_worker_tests PRIVATE rave_engine)
    add_test(NAME rave_inference_worker_tests COMMAND rave_inference_worker_tests)

    add_executable(rave_audio_engine_tests tests/RaveAudioEngineTests.cpp)
    target_link_libraries(rave_audio_engine_tests PRIVATE rave_engine)
    add_test(NAME rave_audio_engine_tests COMMAND rave_audio_engine_tests)

    add_executable(rave_standalone_state_tests tests/StandaloneSessionStateTests.cpp)
    target_link_libraries(rave_standalone_state_tests PRIVATE rave_engine juce::juce_core)
    add_test(NAME rave_standalone_state_tests COMMAND rave_standalone_state_tests)

    add_executable(rave_standalone_preset_coordinator_tests
        tests/StandalonePresetModelCoordinatorTests.cpp)
    target_link_libraries(rave_standalone_preset_coordinator_tests
        PRIVATE rave_engine juce::juce_core)
    add_test(NAME rave_standalone_preset_coordinator_tests
        COMMAND rave_standalone_preset_coordinator_tests)

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
    COMPANY_COPYRIGHT "Copyright (c) 2026 RAVE Performance Instrument contributors"
    BUNDLE_ID "org.raveperformance.instrument"
    ICON_BIG "${RAVE_ICON_BIG}"
    ICON_SMALL "${RAVE_ICON_SMALL}"
)

juce_generate_juce_header(rave_instrument)

target_sources(rave_instrument PRIVATE src/app/Main.cpp)

target_link_libraries(rave_instrument
    PRIVATE
        rave_engine
        rave_icon_resources
        juce::juce_audio_utils
        juce::juce_gui_extra
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

juce_add_plugin(rave_plugin
    COMPANY_NAME "RAVE Performance Instrument"
    PLUGIN_MANUFACTURER_CODE RvAI
    PLUGIN_CODE RvPI
    BUNDLE_ID "org.raveperformance.instrument.plugin"
    PRODUCT_NAME "RAVE Performance Instrument"
    COMPANY_COPYRIGHT "Copyright (c) 2026 RAVE Performance Instrument contributors"
    ICON_BIG "${RAVE_ICON_BIG}"
    ICON_SMALL "${RAVE_ICON_SMALL}"
    FORMATS VST3 AU
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
)

target_sources(rave_plugin PRIVATE
    src/plugin/PluginEditor.cpp
    src/plugin/PluginEditor.h
    src/plugin/PluginProcessor.cpp
    src/plugin/PluginProcessor.h
)

target_include_directories(rave_plugin PRIVATE src)

juce_generate_juce_header(rave_plugin)

target_compile_definitions(rave_plugin
    PRIVATE
        JUCE_USE_CURL=0
        JUCE_WEB_BROWSER=0
        JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(rave_plugin
    PRIVATE
        rave_engine
        rave_icon_resources
        juce::juce_audio_processors
        juce::juce_gui_extra
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

if(APPLE)
    # JUCE puts the generated Icon.icns in each supported bundle and its
    # generated Info.plist. Preserve the canonical SVG as a inspectable bundle
    # resource as well, without adding a runtime file dependency to the UI.
    set_source_files_properties("${RAVE_ICON_SOURCE}"
        PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
    target_sources(rave_instrument PRIVATE "${RAVE_ICON_SOURCE}")
    if(TARGET rave_plugin_VST3)
        target_sources(rave_plugin_VST3 PRIVATE "${RAVE_ICON_SOURCE}")
    endif()
    if(TARGET rave_plugin_AU)
        target_sources(rave_plugin_AU PRIVATE "${RAVE_ICON_SOURCE}")
    endif()

    add_custom_command(TARGET rave_plugin_VST3 POST_BUILD
        COMMAND codesign --force --deep --sign - "$<TARGET_BUNDLE_DIR:rave_plugin_VST3>"
        VERBATIM
    )
    add_custom_command(TARGET rave_plugin_AU POST_BUILD
        COMMAND codesign --force --deep --sign - "$<TARGET_BUNDLE_DIR:rave_plugin_AU>"
        VERBATIM
    )
endif()

if(BUILD_TESTING)
    add_executable(rave_plugin_tests tests/PluginProcessorTests.cpp)
    target_include_directories(rave_plugin_tests PRIVATE src)
    # Force-include the assertion recorder before any JUCE header so JUCE
    # assertions in test-compiled code fail the test instead of only logging.
    target_compile_options(rave_plugin_tests PRIVATE
        "-include${CMAKE_CURRENT_SOURCE_DIR}/tests/JuceTestAssertions.h")
    target_link_libraries(rave_plugin_tests PRIVATE
        rave_plugin
        juce::juce_audio_processors
    )
    add_test(NAME rave_plugin_tests COMMAND rave_plugin_tests)

    if(Torch_FOUND AND Python3_Interpreter_FOUND)
        add_test(
            NAME rave_plugin_model_recall_tests
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_plugin_model_recall_test.py"
                    "$<TARGET_FILE:rave_plugin_tests>"
        )
    endif()
endif()
