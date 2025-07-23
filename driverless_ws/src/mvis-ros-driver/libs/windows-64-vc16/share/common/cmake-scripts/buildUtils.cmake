set(STRIP_COMMAND "strip")

if (CMAKE_CXX_LIBRARY_ARCHITECTURE STREQUAL "aarch64-linux-gnu")
    # search for programs in the build host directories
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

    # for libraries and headers in the target directories
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

    if (EXISTS "$ENV{BTHOME_4_GCC_AARCH64_NONE_LINUX_GNU}")
        set(CMAKE_CROSSCOMPILING_EMULATOR "${CMAKE_CURRENT_LIST_DIR}/../shell-scripts/qemu-aarch64.sh")
        set(STRIP_COMMAND "$ENV{BTHOME_4_GCC_AARCH64_NONE_LINUX_GNU}/aarch64-none-linux-gnu/bin/strip")
    endif ()
endif ()