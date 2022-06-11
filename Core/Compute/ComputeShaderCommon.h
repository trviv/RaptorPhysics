#ifndef COMPUTE_SHADER_COMMON_H
#define COMPUTE_SHADER_COMMON_H

#include <Header/Math.h>
#include <Utils/IOInterface.h>

typedef pair<uint, uint>  uintPair;
typedef vector<uintPair>  uintPairList;
typedef vector<string>    stringArr;

string join(const stringArr* stringList, const char* delim = " ");

string join(const stringArr& stringList, const char* delim = " ");

stringArr unique(const stringArr* stringList, const char* delim = " ");

string deepReadShaderSource(const char* sourceCode, const stringArr* includeFiles = NULL);

struct FunctionArgs
{
  string funcName;
  stringArr args;
};

struct FunctionInfo : public FunctionArgs
{
  size_t funcDeclOffset;
  size_t funcDefOffset;
  size_t funcEndOffset;
  string funcSource;
  string funcHeader;
  size_t argsBegin;
  size_t argsEnd;
};

FunctionInfo parseFunctionInfo(const string& source, size_t startingOffset, bool forwardSearch);

string processAutoArgumentBuffers(string source, unordered_map<string, uintPairList>& kernelNameArgumentBufferMap);

stringArr convertToHLSL(string source, bool keepInputArgs);

vector<FunctionArgs> getKernelArgs(const string& source);

#endif
