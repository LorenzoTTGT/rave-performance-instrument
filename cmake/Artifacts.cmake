find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED)
if(WIN32)
  add_library(rave_vst3_loader MODULE src/plugin/WindowsVst3Loader.cpp)
  set_target_properties(
    rave_vst3_loader PROPERTIES MSVC_RUNTIME_LIBRARY
                                "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  set(RAVE_LOADER_PATH "$<TARGET_FILE:rave_vst3_loader>")
endif()
if(APPLE)
  # JUCE_PLUGIN_ARTEFACT_FILE is an internal property; use known bundle target
  # paths.
  set(RAVE_ARTIFACTS
      "{\"format\":\"VST3\",\"binary\":\"$<TARGET_FILE:rave_plugin_VST3>\",\"path\":\"$<TARGET_BUNDLE_DIR:rave_plugin_VST3>\"},{\"format\":\"AU\",\"binary\":\"$<TARGET_FILE:rave_plugin_AU>\",\"path\":\"$<TARGET_BUNDLE_DIR:rave_plugin_AU>\"},{\"format\":\"Standalone\",\"binary\":\"$<TARGET_FILE:rave_instrument>\",\"path\":\"$<TARGET_BUNDLE_DIR:rave_instrument>\"}"
  )
else()
  set(RAVE_ARTIFACTS
      "{\"format\":\"VST3\",\"binary\":\"$<TARGET_FILE:rave_plugin_VST3>\",\"path\":\"${CMAKE_BINARY_DIR}/rave_plugin_artefacts/$<CONFIG>/VST3/RAVE Performance Instrument.vst3\"},{\"format\":\"Standalone\",\"binary\":\"$<TARGET_FILE:rave_instrument>\",\"path\":\"$<TARGET_FILE:rave_instrument>\"}"
  )
endif()
execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/artifacts.py"
    identity "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE RAVE_SOURCE_IDENTITY COMMAND_ERROR_IS_FATAL ANY
  OUTPUT_STRIP_TRAILING_WHITESPACE)
set(RAVE_MODEL_TESTER "")
if(TARGET rave_torch_backend_tests)
  set(RAVE_MODEL_TESTER "$<TARGET_FILE:rave_torch_backend_tests>")
endif()
file(
  GENERATE
  OUTPUT "${CMAKE_BINARY_DIR}/artifacts-$<CONFIG>.json"
  CONTENT
    "{\"schema\":1,\"version\":\"${PROJECT_VERSION}\",\"system\":\"${CMAKE_SYSTEM_NAME}\",\"architecture\":\"${CMAKE_SYSTEM_PROCESSOR}\",\"configuration\":\"$<CONFIG>\",\"compiler\":\"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\",\"compiler_path\":\"${CMAKE_CXX_COMPILER}\",\"torch_version\":\"${Torch_VERSION}\",\"torch_lib\":\"${TORCH_INSTALL_PREFIX}/lib\",\"source\":\"${CMAKE_CURRENT_SOURCE_DIR}\",\"identity\":${RAVE_SOURCE_IDENTITY},\"loader\":\"${RAVE_LOADER_PATH}\",\"model_tester\":\"${RAVE_MODEL_TESTER}\",\"juce_license\":\"${JUCE_SOURCE_DIR}/LICENSE.md\",\"scanner\":\"$<TARGET_FILE:juce_vst3_helper>\",\"artifacts\":[${RAVE_ARTIFACTS}]}\n"
)
if(BUILD_TESTING)
  add_test(NAME rave_build_workflow_tests
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/BuildWorkflowTests.py")
endif()
