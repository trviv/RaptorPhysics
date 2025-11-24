/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "FontLoader.h"
#include <Utils/IOInterface.h>
#include <imgui/imgui.h>
#include <unordered_map>

static unordered_map<string, ImFont*> fontDictionary;
static unordered_map<string, uint> fontOffset;
static ImWchar glyphRanges[] = {0xF000, (0xF000 + 0x3FF), 0};

void* FontLoader::readFontFile(const char* font, float fontSize, void* fontConfig)
{
  if (font == NULL)
  {
    return NULL;
  }

  if (fontDictionary.find(font) == fontDictionary.end())
  {
    const ushort offset = 0xF000;
    const string fontData = IOInterface::readFile((font+string(".ttf")).c_str());
    const ImWchar start = offset;//static_cast<ImWchar>(offset + fontDictionary.size() * 0x0400);

    fontDictionary[font] = ImGui::GetIO().Fonts->AddFontFromMemoryTTF((void*)fontData.c_str(), (int)fontData.size(), fontSize, (ImFontConfig*)fontConfig, glyphRanges);
    fontOffset[font] = start;
  }
  return fontDictionary[font];
}

void* FontLoader::getFont(const char* font)
{
  return font ? fontDictionary[font] : NULL;
}

ushort FontLoader::getFontOffset(const char* font)
{
  return font ? fontOffset[font] : 0;
}
