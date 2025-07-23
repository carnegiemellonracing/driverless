setupHeaders(
    HEADERS
        Any.hpp
        BackgroundWorker.hpp
        BlockingQueue.hpp
        Convert.hpp
        Concurrency.hpp
        deprecatedwarning.hpp
        SdkHash.hpp
        IdcUriHelper.hpp
        Optional.hpp
        ParameterData.hpp
        ParameterIndex.hpp
        Uri.hpp
        Utils.hpp
        RgbaColor.hpp
        SdkExceptions.hpp
        SharedBuffer.hpp
        SharedBufferStream.hpp
        StatusCodes.hpp
        ThreadContext.hpp
        ThreadSafe.hpp
        ThreadSync.hpp
        ToHex.hpp
        Unconvertable.hpp
        unit.hpp
        Uri.hpp
        Version.hpp
        WinCompatibility.hpp

    GROUP misc
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/crypto/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/defines/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/units/CMakeLists.txt)
