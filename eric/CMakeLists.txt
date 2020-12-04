cmake_minimum_required(VERSION 3.0)
project(programming_party CXX)

find_package(Armadillo)
if (ARMADILLO_FOUND)
  include_directories("${ARMADILLO_INCLUDE_DIRS}")
endif (ARMADILLO_FOUND)

add_library(molecule
  diag.cpp
  utils.hpp
  utils.cpp
  molecule.hpp
  molecule.cpp
  )

set(STUB "h2o_sto3g")

add_custom_command(
  OUTPUT
  "${STUB}_enuc.dat"
  "${STUB}_s.dat"
  "${STUB}_t.dat"
  "${STUB}_v.dat"
  "${STUB}_eri.dat"
  COMMAND "${CMAKE_COMMAND}" -E copy "${programming_party_SOURCE_DIR}/project3/${STUB}_enuc.dat" "${programming_party_BINARY_DIR}/${STUB}_enuc.dat"
  COMMAND "${CMAKE_COMMAND}" -E copy "${programming_party_SOURCE_DIR}/project3/${STUB}_s.dat" "${programming_party_BINARY_DIR}/${STUB}_s.dat"
  COMMAND "${CMAKE_COMMAND}" -E copy "${programming_party_SOURCE_DIR}/project3/${STUB}_t.dat" "${programming_party_BINARY_DIR}/${STUB}_t.dat"
  COMMAND "${CMAKE_COMMAND}" -E copy "${programming_party_SOURCE_DIR}/project3/${STUB}_v.dat" "${programming_party_BINARY_DIR}/${STUB}_v.dat"
  COMMAND "${CMAKE_COMMAND}" -E copy "${programming_party_SOURCE_DIR}/project3/${STUB}_eri.dat" "${programming_party_BINARY_DIR}/${STUB}_eri.dat"
  DEPENDS
  "${programming_party_SOURCE_DIR}/project3/${STUB}_enuc.dat"
  "${programming_party_SOURCE_DIR}/project3/${STUB}_s.dat"
  "${programming_party_SOURCE_DIR}/project3/${STUB}_t.dat"
  "${programming_party_SOURCE_DIR}/project3/${STUB}_v.dat"
  "${programming_party_SOURCE_DIR}/project3/${STUB}_eri.dat"
  )

add_custom_target(inp_p3_${STUB}
  DEPENDS
  "${programming_party_BINARY_DIR}/${STUB}_enuc.dat"
  "${programming_party_BINARY_DIR}/${STUB}_s.dat"
  "${programming_party_BINARY_DIR}/${STUB}_t.dat"
  "${programming_party_BINARY_DIR}/${STUB}_v.dat"
  "${programming_party_BINARY_DIR}/${STUB}_eri.dat"
  )

if (ARMADILLO_FOUND)
  add_executable(project3 project3/project3.cpp)
  add_executable(project8 project8/project8.cpp)
  add_dependencies(project3 inp_p3_${STUB})
  add_dependencies(project8 inp_p3_${STUB})
  target_link_libraries(project3 molecule "${ARMADILLO_LIBRARIES}")
  target_link_libraries(project8 molecule "${ARMADILLO_LIBRARIES}")
endif (ARMADILLO_FOUND)
