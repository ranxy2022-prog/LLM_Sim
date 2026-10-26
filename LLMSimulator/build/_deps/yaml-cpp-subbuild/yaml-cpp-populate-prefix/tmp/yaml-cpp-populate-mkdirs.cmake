# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/xiangyur/projects/projects/DSE/LLM_Sim/LLMSimulator/src/dram/ramulator2/ext/yaml-cpp")
  file(MAKE_DIRECTORY "/home/xiangyur/projects/projects/DSE/LLM_Sim/LLMSimulator/src/dram/ramulator2/ext/yaml-cpp")
endif()
file(MAKE_DIRECTORY
  "/home/xiangyur/projects/projects/DSE/LLM_Sim/LLMSimulator/build/_deps/yaml-cpp-build"
  "/home/xiangyur/projects/projects/DSE/LLM_Sim/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix"
  "/home/xiangyur/projects/projects/DSE/LLM_Sim/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix/tmp"
  "/home/xiangyur/projects/projects/DSE/LLM_Sim/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix/src/yaml-cpp-populate-stamp"
  "/home/xiangyur/projects/projects/DSE/LLM_Sim/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix/src"
  "/home/xiangyur/projects/projects/DSE/LLM_Sim/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix/src/yaml-cpp-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/xiangyur/projects/projects/DSE/LLM_Sim/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix/src/yaml-cpp-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/xiangyur/projects/projects/DSE/LLM_Sim/LLMSimulator/build/_deps/yaml-cpp-subbuild/yaml-cpp-populate-prefix/src/yaml-cpp-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
