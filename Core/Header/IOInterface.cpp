#include "GLClass.h"
#include "IOInterface.h"
#include "ComputeInterface.h"

#include <fstream>
#include <stdio.h>  /* defines FILENAME_MAX */
// #define WINDOWS  /* uncomment this line to use it for windows.*/
#ifdef _WIN32
#include <Pathcch.h>
#else
#include <limits.h>
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
#include <stdarg.h>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"

static unordered_map<string, ImFont*> fontDictionary;
static unordered_map<string, uint> fontOffset;

string getCurrentDir(void)
{
  char *currentPath = new char[2048];
#if   ENV_WIN
  int len = GetModuleFileName(NULL, currentPath, 2047);
#elif ENV_APPLE
  currentPath[0] = NULL;
  const char *executablePath = [[[[[NSProcessInfo processInfo] arguments] objectAtIndex:0] stringByDeletingLastPathComponent] fileSystemRepresentation];
  strcpy(currentPath, executablePath);
  size_t len = strnlen(currentPath, 2047);
  // add an additional / at the end so that directory name does not get deleted
  currentPath[len] = '/';
  currentPath[len + 1] = NULL;
  len += 1;
#else
  ssize_t len = ::readlink("/proc/self/exe", currentPath, 2047);
#endif
  if (len != -1)
  {
    currentPath[len] = '\0';
  }
  logComputeMessage("\nCurrent Execution Path: %s\n", executablePath);
  std::string ret = std::string(currentPath);
  delete[] currentPath;
  return ret.substr(0, ret.find_last_of("\\/"));
}

bool IOInterface::checkFileExist(const char* fileName)
{
  std::string directory = getCurrentDir();
#if __APPLE__ && TARGET_OS_OSX
  return access((directory + "/../Resources/" + fileName).c_str(), F_OK) != -1;
#else
  return access((directory + "/" + fileName).c_str(), F_OK) != -1;
#endif
}

bool IOInterface::checkImageExist(const char* fileName)
{
  return checkFileExist((fileName + string(".png")).c_str());
}

std::string IOInterface::readFile(const char* fileName)
{
  std::string directory = getCurrentDir();
  std::string data;
  std::ifstream file;
#if __APPLE__ && TARGET_OS_OSX
  file.open(directory + "/../Resources/" + fileName, std::ios::binary);
#else
  file.open(directory + "/" + fileName, std::ios::binary);
#endif

  file.seekg(0, std::ios::end);
  data.reserve((size_t)file.tellg());
  file.seekg(0, std::ios::beg);
  data.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  file.close();

  return data;
}

bool IOInterface::readImageFile(const char *nameWithoutExtension, Texture *texture)
{
  int imageWidth = 0;
  int imageHeight = 0;
  int comp;

  string fileData = readFile((nameWithoutExtension+string(".png")).c_str());
  unsigned char* imageData = stbi_load_from_memory((unsigned char*)fileData.c_str(), (int)fileData.size(), &imageWidth, &imageHeight, &comp, 4);

  if (imageData == NULL)
  {
    return false;
  }

  *texture = Texture(TEXTURE_FORMAT_UBYTE);

  texture->init(imageWidth, imageHeight);
  texture->gen((float*)imageData, Texture::COLOR_BUFFER_UCHAR);

  stbi_image_free(imageData);

  return true;
}

bool IOInterface::readImageFile(const char *nameWithoutExtension, Texture *texture, uint resizeWidth, uint resizeHeight)
{
  int imageWidth = 0;
  int imageHeight = 0;
  int comp;

  string fileData = readFile((nameWithoutExtension+string(".png")).c_str());
  unsigned char* imageData = stbi_load_from_memory((unsigned char*)fileData.c_str(), (int)fileData.size(), &imageWidth, &imageHeight, &comp, 4);

  if (imageData == NULL)
  {
    return false;
  }

  *texture = Texture(TEXTURE_FORMAT_UBYTE);

  vector<unsigned char> imageData2;
  imageData2.resize(resizeWidth*resizeHeight*4);
  stbir__resize_arbitrary(NULL, imageData, imageWidth, imageHeight, 0, &imageData2[0], resizeWidth, resizeHeight, 0,
                          0, 0, 1, 1, NULL,
                          4, -1, 0, STBIR_TYPE_UINT8, STBIR_FILTER_DEFAULT, STBIR_FILTER_DEFAULT,
                          STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP, STBIR_COLORSPACE_LINEAR);
  stbi_image_free(imageData);

  texture->init(resizeWidth, resizeHeight);
  texture->gen((float*)&imageData2[0], Texture::COLOR_BUFFER_UCHAR);

  return true;
}

void* IOInterface::readFontFile(const char* font, float fontSize, void* fontConfig)
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
    const ImWchar glyphRanges[] = {start, (ImWchar)(start + 0x3FF), 0};

    fontDictionary[font] = ImGui::GetIO().Fonts->AddFontFromMemoryTTF((void*)fontData.c_str(), (int)fontData.size(), fontSize, (ImFontConfig*)fontConfig, glyphRanges);
    fontOffset[font] = start;
  }
  return fontDictionary[font];
}

void* IOInterface::getFont(const char* font)
{
  return font ? fontDictionary[font] : NULL;
}

ushort IOInterface::getFontOffset(const char* font)
{
  return font ? fontOffset[font] : 0;
}
