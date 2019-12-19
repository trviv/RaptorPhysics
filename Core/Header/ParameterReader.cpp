#include "ParameterReader.h"
#include <sstream>

void ParameterReader::readCSVFile(const char* fileName)
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

  // if bindings exist, set bound values
  if (bindings.size() > 0)
  {
    // for all csv rows
    for (const auto& row : csvData)
    {
      // for all indices in the row
      for (int index = 0; index<row.second.size(); index++)
      {
        const pair<string, int> key = {row.first, index};

        // if binding found
        if (bindings.find(key) != bindings.end())
        {
          setParam(row.first, index, bindings[key].second, bindings[key].first);
        }
      }
    }
  }
}

string ParameterReader::getParamAsString(const string param, const int index)const
{
  if (csvData.find(param) == csvData.end())
  {
    logComputeError("Paramter %s not found!", param.c_str());
  }

  return csvData.at(param)[index];
}

int ParameterReader::getParamAsInt(const string param, const int index)const
{
  if (csvData.find(param) == csvData.end())
  {
    logComputeError("Paramter %s not found!", param.c_str());
  }

  return atoi(csvData.at(param)[index].c_str());
}

bool ParameterReader::getParamAsBool(const string param, const int index)const
{
  if (csvData.find(param) == csvData.end())
  {
    logComputeError("Paramter %s not found!", param.c_str());
  }

  return csvData.at(param)[index] == "true";
}

void ParameterReader::setParam(const string param, const int index, void *address, InputParameterType type)
{
  switch (type)
  {
    case ParameterTypeBool:
      *((bool*)address) = getParamAsBool(param, index);
      break;
    case ParameterTypeInt:
      *((int*)address) = getParamAsInt(param, index);
      break;
    case ParameterTypeString:
      *((string*)address) = getParamAsBool(param, index);
      break;
    default:
      break;
  }
}

void ParameterReader::bindParameter(const string param, const int index, void* address, InputParameterType type)
{
  pair<string, int> key = {param, index};
  if (csvData.find(param) != csvData.end())
  {
    // set parameter value if binding exist
    setParam(param, index, address, type);
  }
  else
  {
    // create new binding if not present
    bindings[key] = pair<InputParameterType, void*>(type, address);
  }
}
