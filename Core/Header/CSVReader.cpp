#include "CSVReader.h"
#include <sstream>

void CSVReader::readFile(const char* fileName)
{
  string fileData = ::readFile(fileName);
  string row, key, word;

  std::istringstream stringStream(fileData);

  while (std::getline(stringStream, row))
  {
    // used for breaking words
    stringstream rowStream(row);

    // read every column data of a row and
    // store it in a string variable, 'word'
    int index = 0;
    while (getline(rowStream, word, ','))
    {
      if (index == 0)
      {
        key = word;
        csvData[key].clear();
      }
      else
      {
        csvData[key].push_back(word);
      }
      index++;
    }
  }
}

string CSVReader::getParamAsString(const string param, const int index)const
{
  if (csvData.find(param) == csvData.end())
  {
    logComputeError("Paramter %s not found!", param.c_str());
  }

  return csvData.at(param)[index];
}

int CSVReader::getParamAsInt(const string param, const int index)const
{
  if (csvData.find(param) == csvData.end())
  {
    logComputeError("Paramter %s not found!", param.c_str());
  }

  return atoi(csvData.at(param)[index].c_str());
}
