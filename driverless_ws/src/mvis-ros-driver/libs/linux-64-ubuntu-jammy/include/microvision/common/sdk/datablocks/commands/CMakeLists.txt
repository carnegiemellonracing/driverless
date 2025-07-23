setupHeaders(
    HEADERS
        Command2010Exporter2010.hpp
        Command2010Importer2010.hpp
        Command2010.hpp
        Command.hpp
        EmptyCommandExporter.hpp

    GROUP datablocks\\commands
    TARGET sdk-devices
)

include(${CMAKE_CURRENT_LIST_DIR}/ecucommands/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/luxcommands/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/miniluxcommands/CMakeLists.txt)
