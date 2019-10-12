mkdir -p build/build_ios
mkdir -p build/build_macos

cd build/build_macos

/Applications/CMake.app/Contents/bin/cmake -GXcode -DCMAKE_SYSTEM_NAME=Darwin ../..

cd ../../build/build_ios

/Applications/CMake.app/Contents/bin/cmake -GXcode -DCMAKE_SYSTEM_NAME=iOS -DCREATE_IOS_PROJECT=true ../..
