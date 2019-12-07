#ifndef CSV_READER_H
#define CSV_READER_H

#include "ComputeInterface.h"
#include <unordered_map>

/*!
@class Class to read and retain values from a CSV file.
*/
class CSVReader
{
  unordered_map<string, vector<string>> csvData;
public:

  void readFile(const char* fileName);

  string getParamAsString(const string param, const int index = 0)const;

  int getParamAsInt(const string param, const int index = 0)const;
};

#endif
