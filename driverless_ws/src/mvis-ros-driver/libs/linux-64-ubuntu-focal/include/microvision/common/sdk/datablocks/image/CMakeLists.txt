setupHeaders(
    HEADERS
        FileFormat.hpp
        ImageBuffer.hpp
        Image.hpp
        ImageExporter2403.hpp
        ImageExporter2404.hpp
        ImageExporter2405.hpp
        ImageFormat.hpp
        ImageImporter2403.hpp
        ImageImporter2404.hpp
        ImageImporter2405.hpp
        PixelFormat.hpp

    GROUP datablocks\\image
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
