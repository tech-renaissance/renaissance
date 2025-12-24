add_subdirectory(base)
add_subdirectory(data)


add_library(renaissance INTERFACE)

target_link_libraries(renaissance INTERFACE
    base
    data
)

target_include_directories(renaissance INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_compile_definitions(tech_renaissance INTERFACE
    TR_VERSION_MAJOR=3
    TR_VERSION_MINOR=6
    TR_VERSION_PATCH=0
    TR_VERSION="3.6.0"
)

message(STATUS "Source modules configured successfully")
