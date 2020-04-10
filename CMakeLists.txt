cmake_minimum_required(VERSION 3.10)
project(Cyron)

set(CMAKE_CXX_STANDARD 17)

#find_package(Boost COMPONENTS filesystem program_options system locale REQUIRED)
find_package(Boost REQUIRED)
#find_package(Torch REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

add_executable(Cyron main.cpp src/exploration.cpp src/exploration.h src/network.cpp src/network.h)
target_link_libraries(Cyron LINK_PUBLIC ${Boost_LIBRARIES})
#target_link_libraries(Cyron LINK_PUBLIC ${TORCH_LIBRARIES})