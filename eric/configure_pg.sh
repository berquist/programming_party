#!/usr/bin/env bash

# run me a single level *above* the build directory!

mkdir -p build
cd build
rm -rf ./*

cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -pg" \
    ..
