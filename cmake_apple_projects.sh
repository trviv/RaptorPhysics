#!/bin/bash

mkdir -p build/build_ios
mkdir -p build/build_macos

echo "Signing Identities:"
/usr/bin/env xcrun security find-identity -v -p codesigning
identities=$(/usr/bin/env xcrun security find-identity -v -p codesigning)
IFS=$'\n' read -ra identities <<< "$identities"
IFS=' ' read -ra identities <<< "$identities"
identity=${identities[1]}
echo "Taking Identity: " $identity
echo

cd build/build_macos

/Applications/CMake.app/Contents/bin/cmake -GXcode -DCMAKE_SYSTEM_NAME=Darwin -DSIGNING_IDENTITY=$identity ../..

cd ../../build/build_ios

/Applications/CMake.app/Contents/bin/cmake -GXcode -DCMAKE_SYSTEM_NAME=iOS -DCREATE_IOS_PROJECT=true -DSIGNING_IDENTITY=$identity ../..
