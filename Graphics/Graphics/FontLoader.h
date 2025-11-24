/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef FONT_LOADER_H
#define FONT_LOADER_H

#include <Header/Math.h>

class FontLoader
{
public:

  static void* readFontFile(const char* font, float fontSize, void* fontConfig);

  static void* getFont(const char* font);

  static ushort getFontOffset(const char* font);

};

#endif
