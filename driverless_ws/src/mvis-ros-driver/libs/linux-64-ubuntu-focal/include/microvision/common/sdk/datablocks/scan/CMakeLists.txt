setupHeaders(
    HEADERS
        ResolutionInfo.hpp
        Scan.hpp
        ScanExporter2209.hpp
        ScanImporter2202.hpp
        ScanImporter2205.hpp
        ScanImporter2208.hpp
        ScanImporter2209.hpp
        ScanImporter2321.hpp
        ScannerInfo.hpp
        ScanPoint.hpp

    GROUP datablocks\\scan
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
