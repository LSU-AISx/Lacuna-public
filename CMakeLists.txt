cmake_minimum_required(VERSION 3.20)
project(lacuna_core VERSION 0.1.0 LANGUAGES C)

set(LACUNA_REAL_BITS "64" CACHE STRING "Model arithmetic width")
set(LACUNA_TIME_BITS "64" CACHE STRING "Event timestamp width")
if((NOT LACUNA_REAL_BITS MATCHES "^(16|32|64)$") OR
   (NOT LACUNA_TIME_BITS MATCHES "^(16|32|64)$") OR
   (LACUNA_REAL_BITS STREQUAL "64" AND NOT LACUNA_TIME_BITS STREQUAL "64") OR
   ((LACUNA_REAL_BITS STREQUAL "16" OR LACUNA_TIME_BITS STREQUAL "16") AND
    NOT LACUNA_REAL_BITS STREQUAL LACUNA_TIME_BITS))
    message(FATAL_ERROR
        "Unsupported LACUNA_REAL_BITS=${LACUNA_REAL_BITS}, "
        "LACUNA_TIME_BITS=${LACUNA_TIME_BITS}. "
        "Supported numeric profiles are model/time bits 64/64, 32/64, 32/32, and strict 16/16."
    )
endif()
if(LACUNA_TIME_BITS LESS_EQUAL 32)
    set(LACUNA_PROFILING_DEFAULT OFF)
else()
    set(LACUNA_PROFILING_DEFAULT ON)
endif()
option(LACUNA_ENABLE_PROFILING "Enable native kernel wall-clock timing" ${LACUNA_PROFILING_DEFAULT})
if(LACUNA_TIME_BITS LESS_EQUAL 32 AND LACUNA_ENABLE_PROFILING)
    message(FATAL_ERROR "Strict reduced-precision execution requires LACUNA_ENABLE_PROFILING=OFF.")
endif()

add_library(
    lacuna_core SHARED
    c/src/lacuna.c
    c/src/network.c
    c/src/compiled_graph_io.c
    c/src/expr.c
    c/src/codec.c
)
target_include_directories(lacuna_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/c/include)
target_compile_definitions(lacuna_core PUBLIC
    LACUNA_REAL_BITS=${LACUNA_REAL_BITS} LACUNA_TIME_BITS=${LACUNA_TIME_BITS}
    LACUNA_ENABLE_PROFILING=$<BOOL:${LACUNA_ENABLE_PROFILING}>
)
target_compile_features(lacuna_core PRIVATE c_std_99)
target_compile_options(lacuna_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(LACUNA_REAL_BITS LESS_EQUAL 32)
    target_compile_options(lacuna_core PRIVATE -fno-fast-math -ffp-contract=off)
endif()
if(LACUNA_TIME_BITS STREQUAL "32")
    target_compile_options(lacuna_core PRIVATE -Wdouble-promotion)
endif()

if(LACUNA_REAL_BITS STREQUAL "16")
    target_sources(lacuna_core PRIVATE c/src/half_math.c)
    target_compile_options(lacuna_core PUBLIC -fno-fast-math -ffp-contract=off)
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set(LACUNA_HALF_EVAL_FLAG -fexcess-precision=16)
    elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(LACUNA_HALF_EVAL_FLAG -ffp-eval-method=source)
    else()
        message(FATAL_ERROR "Strict float16 requires an audited GCC or Clang toolchain.")
    endif()
    target_compile_options(lacuna_core PUBLIC ${LACUNA_HALF_EVAL_FLAG})
    # This shim transports half scalars as bits for the host ctypes interface.
    option(LACUNA_BUILD_HALF_HOST "Build the optional host half transport" ON)
    if(LACUNA_BUILD_HALF_HOST)
        add_library(lacuna_half_host SHARED c/src/half_host_bridge.c)
        target_include_directories(lacuna_half_host PRIVATE c/include)
        target_compile_definitions(lacuna_half_host PRIVATE
            LACUNA_REAL_BITS=16 LACUNA_TIME_BITS=16 LACUNA_ENABLE_PROFILING=0
        )
        target_compile_options(lacuna_half_host PRIVATE
            -Wall -Wextra -Wpedantic -Werror -fno-fast-math -ffp-contract=off
            ${LACUNA_HALF_EVAL_FLAG}
        )
    endif()
endif()

if(NOT WIN32 AND NOT LACUNA_REAL_BITS STREQUAL "16")
    target_link_libraries(lacuna_core PRIVATE m)
endif()

option(LACUNA_BUILD_TARGET_BINDER "Build the host target constant binder" ON)
if(LACUNA_BUILD_TARGET_BINDER)
    add_library(lacuna_target_bind SHARED c/src/target_bind.c)
    target_include_directories(lacuna_target_bind PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/c/include)
    target_compile_features(lacuna_target_bind PRIVATE c_std_99)
    target_compile_options(lacuna_target_bind PRIVATE
        -Wall -Wextra -Wpedantic -Werror -fno-fast-math -ffp-contract=off
    )
    if(NOT WIN32)
        target_link_libraries(lacuna_target_bind PRIVATE m)
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    if(LACUNA_REAL_BITS STREQUAL "64")
    add_executable(lacuna_core_tests tests/c/test_scalar_lif.c)
    target_link_libraries(lacuna_core_tests PRIVATE lacuna_core)
    if(NOT WIN32)
        # The test translation unit calls exp() directly, so it must declare
        # its own libm dependency rather than relying on the shared library's
        # private link interface.
        target_link_libraries(lacuna_core_tests PRIVATE m)
    endif()
    # The native suite uses assert() as its executable test mechanism. Keep
    # assertions enabled even when the library itself is built in Release mode.
    target_compile_options(
        lacuna_core_tests PRIVATE -Wall -Wextra -Wpedantic -Werror -UNDEBUG
    )
    add_test(NAME lacuna_core_tests COMMAND lacuna_core_tests)

    add_executable(lacuna_numeric_type_tests tests/c/test_numeric_types.c)
    target_link_libraries(lacuna_numeric_type_tests PRIVATE lacuna_core)
    if(NOT WIN32)
        target_link_libraries(lacuna_numeric_type_tests PRIVATE m)
    endif()
    target_compile_features(lacuna_numeric_type_tests PRIVATE c_std_99)
    target_compile_options(
        lacuna_numeric_type_tests PRIVATE -Wall -Wextra -Wpedantic -Werror -UNDEBUG
    )
    add_test(NAME lacuna_numeric_type_tests COMMAND lacuna_numeric_type_tests)
    endif()

    if(NOT LACUNA_REAL_BITS STREQUAL "16")
        add_executable(lacuna_native_precision_tests tests/c/test_native_precision.c)
        target_link_libraries(lacuna_native_precision_tests PRIVATE lacuna_core)
        if(NOT WIN32)
            target_link_libraries(lacuna_native_precision_tests PRIVATE m)
        endif()
        target_compile_options(lacuna_native_precision_tests PRIVATE
            -Wall -Wextra -Wpedantic -Werror -UNDEBUG
        )
        add_test(NAME lacuna_native_precision_tests COMMAND lacuna_native_precision_tests)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/c/test_image_precision.c")
        add_executable(lacuna_image_precision_tests tests/c/test_image_precision.c)
        target_link_libraries(lacuna_image_precision_tests PRIVATE lacuna_core)
        if(NOT WIN32)
            target_link_libraries(lacuna_image_precision_tests PRIVATE m)
        endif()
        target_compile_options(lacuna_image_precision_tests PRIVATE
            -Wall -Wextra -Wpedantic -Werror -UNDEBUG
        )
        add_test(NAME lacuna_image_precision_tests COMMAND lacuna_image_precision_tests)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/c/test_codec_precision.c")
        add_executable(lacuna_codec_precision_tests tests/c/test_codec_precision.c)
        target_link_libraries(lacuna_codec_precision_tests PRIVATE lacuna_core)
        if(NOT WIN32)
            target_link_libraries(lacuna_codec_precision_tests PRIVATE m)
        endif()
        target_compile_options(lacuna_codec_precision_tests PRIVATE
            -Wall -Wextra -Wpedantic -Werror -UNDEBUG
        )
        add_test(NAME lacuna_codec_precision_tests COMMAND lacuna_codec_precision_tests)
    endif()

    if(LACUNA_BUILD_TARGET_BINDER)
        add_executable(lacuna_target_bind_tests tests/c/test_target_bind.c)
        target_link_libraries(lacuna_target_bind_tests PRIVATE lacuna_target_bind)
        if(NOT WIN32)
            target_link_libraries(lacuna_target_bind_tests PRIVATE m)
        endif()
        target_compile_features(lacuna_target_bind_tests PRIVATE c_std_99)
        target_compile_options(lacuna_target_bind_tests PRIVATE
            -Wall -Wextra -Wpedantic -Werror -UNDEBUG -fno-fast-math -ffp-contract=off
        )
        add_test(NAME lacuna_target_bind_tests COMMAND lacuna_target_bind_tests)
    endif()

    if(LACUNA_REAL_BITS STREQUAL "16")
        add_executable(lacuna_half_math_tests tests/c/test_half_math.c)
        target_link_libraries(lacuna_half_math_tests PRIVATE lacuna_core)
        target_compile_options(lacuna_half_math_tests PRIVATE
            -Wall -Wextra -Wpedantic -Werror -UNDEBUG
        )
        add_test(NAME lacuna_half_math_tests COMMAND lacuna_half_math_tests)

        add_executable(lacuna_half_runtime_tests tests/c/test_half_runtime.c)
        target_link_libraries(lacuna_half_runtime_tests PRIVATE lacuna_core)
        target_compile_options(lacuna_half_runtime_tests PRIVATE
            -Wall -Wextra -Wpedantic -Werror -UNDEBUG
        )
        add_test(NAME lacuna_half_runtime_tests COMMAND lacuna_half_runtime_tests)
    endif()

    add_executable(lacuna_hazard_precision_tests tests/c/test_hazard_precision.c)
    target_link_libraries(lacuna_hazard_precision_tests PRIVATE lacuna_core)
    if(NOT WIN32)
        target_link_libraries(lacuna_hazard_precision_tests PRIVATE m)
    endif()
    target_compile_options(lacuna_hazard_precision_tests PRIVATE
        -Wall -Wextra -Wpedantic -Werror -UNDEBUG
    )
    add_test(NAME lacuna_hazard_precision_tests COMMAND lacuna_hazard_precision_tests)
endif()
