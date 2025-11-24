/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef IOINTERFACE_H
#define IOINTERFACE_H

#include <Header/Math.h>
#include <vector>
#include <unordered_map>


class IOInterface
{
public:

  static string getPath(const char* fileName);

  static bool checkFileExist(const char* fileName);

  static string readFile(const char* fileName);

  static void writeFile(const char* fileName, const char* fileData, size_t fileDataSize);

  static vector<char> readByteFile(const char* fileName);

};

#endif
