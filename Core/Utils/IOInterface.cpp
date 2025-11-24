/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "IOInterface.h"
#include <Compute/ComputeInterface.h>

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


string getCurrentDir(void)
{
  char *currentPath = new char[2048];
#if   ENV_WIN
  int len = GetModuleFileName(NULL, currentPath, 2047);
  const char *executablePath = currentPath;
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
  const char *executablePath = currentPath;
#endif
  if (len != -1)
  {
    currentPath[len] = '\0';
  }
  std::string ret = std::string(currentPath);
  delete[] currentPath;
  return ret.substr(0, ret.find_last_of("\\/"));
}

string IOInterface::getPath(const char* fileName)
{
  const std::string directory = getCurrentDir();
#if __APPLE__
#if TARGET_OS_OSX
  return directory + "/../Resources/" + fileName;
#else
  return directory + "/" + fileName;
#endif
#else
  return directory + "/" + fileName;
#endif
}

bool IOInterface::checkFileExist(const char* fileName)
{
  const std::string path = getPath(fileName);
#if __APPLE__
  return access(path.c_str(), F_OK) != -1;
#else
  struct stat buffer;
  return stat(path.c_str(), &buffer) == 0;
#endif
}

std::string IOInterface::readFile(const char* fileName)
{
  const std::string path = getPath(fileName);
  std::string data;

  std::ifstream file;
  file.open(path, std::ios::binary);
  file.seekg(0, std::ios::end);
  data.reserve((size_t)file.tellg());
  file.seekg(0, std::ios::beg);
  data.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  file.close();

  return data;
}

void IOInterface::writeFile(const char* fileName, const char* fileData, size_t fileDataSize)
{
  std::string path = getPath(fileName);

  std::ofstream file;
  file.open(path.c_str(), std::ios::binary);
  file.write(fileData, fileDataSize);

  file.close();
}

vector<char> IOInterface::readByteFile(const char* fileName)
{
  const std::string path = getPath(fileName);
  vector<char> data;

  std::ifstream file;
  file.open(path, std::ios::binary);
  file.seekg(0, std::ios::end);
  data.reserve((size_t)file.tellg());
  file.seekg(0, std::ios::beg);
  data.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  file.close();

  return data;
}

