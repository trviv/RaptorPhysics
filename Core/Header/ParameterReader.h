#ifndef PARAMETER_READER_H
#define PARAMETER_READER_H

#include "ComputeInterface.h"
#include <unordered_map>

enum InputParameterType
{
  ParameterTypeNone,
  ParameterTypeBool,
  ParameterTypeInt,
  ParameterTypeFloat,
  ParameterTypeString
};

/*!
@class Class to read and retain parameteres from a file.
*/
class ParameterReader
{
  unordered_map<string, vector<string>> csvData;
  map<pair<string, int>, pair<InputParameterType, void*>> bindings;

  void setParam(const string param, const int index, void* address, InputParameterType type);

public:

  void readCSVFile(const char* fileName);

  string getParamAsString(const string param, const int index = 0)const;

  bool getParamAsBool(const string param, const int index = 0)const;

  int getParamAsInt(const string param, const int index = 0)const;

  float getParamAsFloat(const string param, const int index = 0)const;

  void bindParameter(const string param, const int index = 0, void* address = NULL, InputParameterType type = ParameterTypeNone);
};

#endif
