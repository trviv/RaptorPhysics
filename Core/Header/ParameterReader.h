#ifndef PARAMETER_READER_H
#define PARAMETER_READER_H

#include "IOInterface.h"
#include "../Vector/Real3.h"
#include "ComputeInterface.h"

enum InputParameterType
{
  ParameterTypeNone,
  ParameterTypeBool,
  ParameterTypeInt,
  ParameterTypeFloat,
  ParameterTypeFloat3,
  ParameterTypeString,
  ParameterTypeMax
};

typedef vector<int> Int2;

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

  Int2 getParamAsInt2(const string param)const;

  float getParamAsFloat(const string param, const int index = 0)const;

  Real3 getParamAsFloat3(const string param)const;

  void bindParameter(const string param, void* address, InputParameterType type, const int index = 0);
};

#endif
