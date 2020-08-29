#include "ParameterReader.h"
#include <sstream>

void ParameterReader::readCSVFile(const char* fileName)
{
  string fileData = IOInterface::readFile(fileName);
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
        word.erase(std::remove_if(word.begin(), word.end(), ::isspace), word.end());
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

bool ParameterReader::getParamAsBool(const string param, const int index)const
{
  return getParamAsString(param, index) == "true";
}

int ParameterReader::getParamAsInt(const string param, const int index)const
{
  return atoi(getParamAsString(param, index).c_str());
}

Int2 ParameterReader::getParamAsInt2(const string param)const
{
  vector<int> ret(2);
  ret[0] = atoi(getParamAsString(param, 0).c_str());
  ret[1] = atoi(getParamAsString(param, 1).c_str());
  return ret;
}

float ParameterReader::getParamAsFloat(const string param, const int index)const
{
  return atof(getParamAsString(param, index).c_str());
}

Real3 ParameterReader::getParamAsFloat3(const string param)const
{
  return Real3(getParamAsFloat(param, 0), getParamAsFloat(param, 1), getParamAsFloat(param, 2));
}

string ParameterReader::getParamAsString(const string param, const int index)const
{
  if (csvData.find(param) == csvData.end())
  {
    logComputeError("Paramter %s not found!", param.c_str());
  }

  return csvData.at(param)[index];
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
    case ParameterTypeFloat:
      *((float*)address) = getParamAsFloat(param, index);
      break;
    case ParameterTypeFloat3:
      *((Real3*)address) = getParamAsFloat3(param);
      break;
    case ParameterTypeString:
      *((string*)address) = getParamAsString(param, index);
      break;
    default:
      break;
  }
}

void ParameterReader::bindParameter(const string param, void* address, InputParameterType type, const int index)
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
