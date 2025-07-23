setupHeaders(
    HEADERS
        BagDataStreamReader.hpp
        BagDataStreamWriter.hpp
        BagRecordChunkStreamReader.hpp
        BagRecordChunkStreamWriter.hpp
        BagRecordStreamReader.hpp
        BagRecordStreamWriter.hpp
        BagStreamBase.hpp
        BagStreamReaderBase.hpp
        BagStreamWriterBase.hpp

    GROUP io\\bag
    TARGET sdk-data
)

include(${CMAKE_CURRENT_LIST_DIR}/package/CMakeLists.txt)