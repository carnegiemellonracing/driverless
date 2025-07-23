set(JPEG_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../libs/windows-64-vc16")
set(JPEG_LIBRARY "${JPEG_ROOT_DIR}/lib/jpeg.lib")

set(JPEG_FOUND TRUE)
set(JPEG_LIBRARIES ${JPEG_LIBRARY})
set(JPEG_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../libs/windows-64-vc16/include")

if(JPEG_FOUND)
    if(NOT TARGET JPEG::JPEG)
        add_library(JPEG::JPEG UNKNOWN IMPORTED)
        set_target_properties(JPEG::JPEG PROPERTIES
            IMPORTED_LOCATION "${JPEG_LIBRARY}"
        )
    endif()
endif()
