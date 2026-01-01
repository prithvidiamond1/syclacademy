#[[
  SYCL Academy (c)

  SYCL Academy is licensed under a Creative Commons Attribution-ShareAlike 4.0
  International License.

  You should have received a copy of the license along with this work.  If not,
  see <http://creativecommons.org/licenses/by-sa/4.0/>.
]]

if (CMAKE_HOST_WIN32)
# need CMake 3.25.0+ for IntelLLVM support of target link properties on Windows
cmake_minimum_required(VERSION 3.25)
else()
# CMake 3.23.5 is the minimum recommended for IntelLLVM on Linux
cmake_minimum_required(VERSION 3.23.5)
endif()

project(What_Is_SYCL LANGUAGES CXX)

# Export compile_commands.json for clangd to use
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find the Intel DPC++ configuration package
find_package(IntelSYCL REQUIRED)

add_executable(solution solution.cpp)
add_sycl_to_target(TARGET solution SOURCES solution.cpp)

add_executable(source source.cpp)
add_sycl_to_target(TARGET source SOURCES source.cpp)