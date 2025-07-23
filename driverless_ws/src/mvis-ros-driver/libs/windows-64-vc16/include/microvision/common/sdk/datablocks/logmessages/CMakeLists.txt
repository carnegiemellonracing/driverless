setupHeaders(
    HEADERS
        LogMessage.hpp
        LogMessageExporter.hpp
        LogMessageExporter6400.hpp
        LogMessageExporter6410.hpp
        LogMessageExporter6420.hpp
        LogMessageExporter6430.hpp
        LogMessageImporter.hpp
        LogMessageImporter6400.hpp
        LogMessageImporter6410.hpp
        LogMessageImporter6420.hpp
        LogMessageImporter6430.hpp

    GROUP datablocks\\logmessages
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
