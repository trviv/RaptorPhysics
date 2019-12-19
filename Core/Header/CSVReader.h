#ifndef CSV_READER_H
#define CSV_READER_H

#include "ComputeInterface.h"
#include <unordered_map>

enum InputParameterType
{
  ParameterTypeNone,
  ParameterTypeInt,
  ParameterTypeString,
  ParameterTypeBool
};

/*!
@class Class to read and retain values from a CSV file.
*/
class CSVReader
{
  unordered_map<string, vector<string>> csvData;
  map<pair<string, int>, pair<InputParameterType, void*>> bindings;

  void setParam(const string param, const int index, void* address, InputParameterType type);

public:

  void readFile(const char* fileName);

  string getParamAsString(const string param, const int index = 0)const;

  int getParamAsInt(const string param, const int index = 0)const;

  bool getParamAsBool(const string param, const int index = 0)const;

  void bindParameter(const string param, const int index = 0, void* address = NULL, InputParameterType type = ParameterTypeNone);
};

#endif
