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

#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_internal.h"

class Texture;

class IOInterface
{
public:

  static string getPath(const char* fileName);

  static bool checkFileExist(const char* fileName);

  static bool checkImageExist(const char* fileName);

  static string readFile(const char* fileName);

  static void writeFile(const char* fileName, const char* fileData, size_t fileDataSize);

  static vector<char> readByteFile(const char* fileName);

  static bool readImageFile(const char *nameWithoutExtension, Texture *texture);

  static bool readImageFile(const char *nameWithoutExtension, Texture *texture, uint resizeWidth, uint resizeHeight);

  static void* readFontFile(const char* font, float fontSize, void* fontConfig);

  static void* getFont(const char* font);

  static ushort getFontOffset(const char* font);

};

#endif
