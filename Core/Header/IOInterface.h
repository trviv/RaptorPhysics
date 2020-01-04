#ifndef IOINTERFACE_H
#define IOINTERFACE_H

#include "Root.h"
#include "Math.h"
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

  static bool checkFileExist(const char* fileName);

  static bool checkImageExist(const char* fileName);

  static string readFile(const char* fileName);

  static bool readImageFile(const char *nameWithoutExtension, Texture *texture);

  static bool readImageFile(const char *nameWithoutExtension, Texture *texture, uint resizeWidth, uint resizeHeight);

  static void* readFontFile(const char* font, float fontSize, void* fontConfig);

  static void* getFont(const char* font);

  static ushort getFontOffset(const char* font);

};

#endif
