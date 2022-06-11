#include "ComputeShaderCommon.h"
#include <unordered_set>
#include "ComputeInterface.h"

const char* kernelString = "Kernel ";

string trim(const string& str, const char* delim = " ")
{
  size_t first = str.find_first_not_of(delim);
  if (string::npos == first)
  {
    return str;
  }
  return str.substr(first, (str.find_last_not_of(delim) - first + 1));
}

bool isComment(const string& input, const size_t startIndex)
{
  // returns true when / is succeeded by / or *
  return input[startIndex] == '/' && (startIndex+1) < input.size() && (input[startIndex+1] == '/' || input[startIndex+1] == '*');
}

size_t skipComment(const string& input, size_t startIndex)
{
  startIndex++;
  if (input[startIndex] == '/')
  {
    // go till next line in case of //
    while (startIndex < input.size() && input[startIndex] != '\n')
    {
      startIndex++;
    }
  }

  if (startIndex < input.size() && input[startIndex] == '*')
  {
    // find first */ in case of /*
    while ((startIndex < input.size() && input[startIndex] != '*') || ((startIndex+1) < input.size() && input[startIndex+1] != '/'))
    {
      startIndex++;
    }
  }

  return startIndex;
}

stringArr tokenize(const string& input, const string delimiter = " \t\r\n")
{
  stringArr tokens;
  string temp;
  size_t curr = 0;
  size_t prev = 0;
  vector<char> bracketStack;
  const string openBrackets   = "({[";
  const string closeBrackets  = ")}]";

  for (uint curr=0; curr<input.size(); curr++)
  {
    const char c = input[curr];

    if (delimiter.find(c) != delimiter.npos)
    {
      if (bracketStack.size())
      {
        continue;
      }

      temp = trim(input.substr(prev, curr - prev));
      if (!temp.empty())
      {
        tokens.push_back(temp);
      }
      prev = curr+1;
    }

    size_t openIndex = openBrackets.find(c);
    if (openIndex != openBrackets.npos)
    {
      bracketStack.push_back(closeBrackets[openIndex]);
    }
    else if (closeBrackets.find(c) != openBrackets.npos)
    {
      if (!bracketStack.empty() && bracketStack.back() == c)
      {
        bracketStack.pop_back();
      }
    }
  }

  temp = trim(input.substr(prev, curr - prev));
  if (!temp.empty())
  {
    tokens.push_back(temp);
  }

  return tokens;
}

string join(const stringArr* stringList, const char* delim)
{
  string ret;

  if (!stringList || !stringList->size()) return  ret;

  ret = stringList->at(0);

  for (int i=1; i<stringList->size(); i++)
  {
    ret += delim + stringList->at(i);
  }

  return ret;
}

string join(const stringArr& stringList, const char* delim)
{
  return join(&stringList, delim);
}

stringArr unique(const stringArr* stringList, const char* delim)
{
  stringArr ret;

  if (!stringList || !stringList->size()) return  ret;

  ret = *stringList;
  sort(ret.begin(), ret.end());

  int endPos = 0;
  for (int i=1; i<ret.size(); i++)
  {
    if (ret[endPos] == ret[i]) continue;

    ret[endPos++] = ret[i-1];
  }

  return ret;
}

size_t findEndOfFunction(const string& input, const size_t startIndex, const char item)
{
  vector<char> bracketStack;
  const string openBrackets   = "({[";
  const string closeBrackets  = ")}]";

  for (size_t curr=startIndex; curr<input.size(); curr++)
  {
    const char c = input[curr];

    if (item == c)
    {
      if (bracketStack.size() == 0)
      {
        return curr;
      }
    }

    if (isComment(input, curr))
    {
      curr = skipComment(input, curr);
      continue;
    }

    const size_t openIndex = openBrackets.find(c);
    if (openIndex != openBrackets.npos)
    {
      bracketStack.push_back(closeBrackets[openIndex]);
    }
    else if (closeBrackets.find(c) != openBrackets.npos)
    {
      if (!bracketStack.empty() && bracketStack.back() == c)
      {
        bracketStack.pop_back();
      }
    }
  }

  return input.npos;
}

string deepReadShaderSource(const char* sourceCode, const stringArr* includeFiles)
{
  string finalSource;

  stack<bool>   sourceSeen;
  stack<string> sources;
  stack<string> sourceName;

  unordered_set<string> processedFiles;

  sourceSeen.push(false);
  sources.push(sourceCode);
  sourceName.push("Base Shader");

  for (uint i = 0; includeFiles && i < includeFiles->size(); i++)
  {
    sourceSeen.push(false);
    sources.push(IOInterface::readFile((*includeFiles)[i].c_str()));
    sourceName.push((*includeFiles)[i].c_str());
  }

  const string includeString = "#include";

  // loop while there are remaining source
  while (sources.size())
  {
    // if osurce processed, add it to the final shader
    if (sourceSeen.top())
    {
      logComputeMessage("Including header: %s", sourceName.top().c_str());
      finalSource += sources.top();
      sourceSeen.pop();
      sources.pop();
      sourceName.pop();
      continue;
    }

    size_t startPos = 0;
    stringArr includeList;

    string& localSource = sources.top();
    sourceSeen.top() = true;

    do
    {
      size_t currPos = localSource.find(includeString, startPos);

      if (currPos == localSource.npos) break;

      // if valid include
      if (currPos < 2 || localSource[currPos - 1] != '/')
      {
        string includeLine = localSource.substr(currPos, localSource.find("\n", currPos) - currPos);
        string includeName;

        stringArr tokens = tokenize(includeLine, "\"");
        if (tokens.size() == 0)
        {
          tokens = tokenize(includeLine, "<");
          includeName = tokenize(tokens.back(), ">")[0];
        }
        else
        {
          includeName = tokens.back();
        }

        // modify the source to commend out the include
        localSource[currPos]   = '/';
        localSource[currPos+1] = '/';

        // add include to list of includes
        includeList.push_back(includeName);
      }

      startPos = currPos + includeString.size();
    }
    while (true);

    // add includes found in the source to stack
    while (includeList.size())
    {
      // if file exist and is not seen yet add it to the stack
      if (processedFiles.count(includeList.back()) == 0 && IOInterface::checkFileExist(includeList.back().c_str()))
      {
        sources.push(IOInterface::readFile(includeList.back().c_str()));
        sourceName.push(includeList.back().c_str());
        sourceSeen.push(false);
        processedFiles.insert(includeList.back());
      }
      includeList.pop_back();
    }
  }

  return finalSource;
}

FunctionInfo parseFunctionInfo(const string& source, size_t startingOffset, bool forwardSearch)
{
  FunctionInfo ret;

  ret.funcDeclOffset = forwardSearch ? source.find(kernelString, startingOffset) : source.rfind(kernelString, startingOffset);
  if (ret.funcDeclOffset == source.npos) return ret;

  while (true)
  {
    size_t newLine = source.rfind("\n", ret.funcDeclOffset);
    string currentLine = source.substr(newLine, ret.funcDeclOffset - newLine);

    if (currentLine.find("#define") != currentLine.npos)
    {
      ret.funcDeclOffset = forwardSearch ? source.find(kernelString, ret.funcDeclOffset + sizeof(kernelString))
                                         : source.rfind(kernelString, ret.funcDeclOffset + sizeof(kernelString));

      if (ret.funcDeclOffset == source.npos) return ret;
    }
    else
    {
      break;
    }
  }

  // go till the beginning of kernel definitions
  ret.funcDefOffset = source.find("{", ret.funcDeclOffset);
  ret.funcEndOffset = findEndOfFunction(source, ret.funcDefOffset + 1, '}');
  ret.funcSource    = source.substr(ret.funcDeclOffset, ret.funcEndOffset - ret.funcDeclOffset + 1);
  ret.funcHeader    = source.substr(ret.funcDeclOffset, ret.funcDefOffset - ret.funcDeclOffset);

  // arguments declaration begin and end
  ret.argsBegin = ret.funcHeader.find('(');
  ret.argsEnd   = ret.funcHeader.rfind(')');
  ret.funcName  = tokenize(ret.funcHeader.substr(0, ret.argsBegin)).back();

  string tempArg = ret.funcHeader.substr(ret.argsBegin + 1, ret.argsEnd - ret.argsBegin - 1);

  size_t ptr = 0;
  ptr = tempArg.find("KERNEL_GLOBAL_ARGUMENTS");
  if (ptr != tempArg.npos) tempArg.insert(ptr, ",");
  ptr = tempArg.find("KERNEL_THREAD_ARGUMENTS");
  if (ptr != tempArg.npos) tempArg.insert(ptr, ",");
  ptr = tempArg.find("KERNEL_THREADGROUP_ARGUMENTS");
  if (ptr != tempArg.npos) tempArg.insert(ptr, ",");

  ret.args      = tokenize(tempArg, ",");

  return ret;
}

inline string generateNewKernelHeader(const string& oldKernelHeader,
                                      const vector<pair<string, stringArr>>& structData,
                                      const uintPairList& structPos,
                                      bool useDefines = true)
{
  string newKernelHeader;

  // declare structure before kernel declaration
  for (const auto &s : structData)
  {
    newKernelHeader.append("typedef struct "+s.first+"{\n");
    for (uint i=0; i<s.second.size(); i++)
    {
      newKernelHeader.append("  "+s.second[i]+" [[ id ("+to_string(i)+") ]];\n");
    }
    newKernelHeader.append("} "+s.first+";\n\n");
  }

  // copy kernel arguments before arg buffer
  newKernelHeader.append(oldKernelHeader, 0, structPos[0].first);

  // place argument buffer struct and members in kernel declaration
  for (uint i=0; i<structPos.size(); i++)
  {
    newKernelHeader.append("Const "+structData[i].first+" *"+structData[i].first+"_args,\n");
  }
  newKernelHeader.pop_back();
  newKernelHeader.pop_back();

  // copy kernel arguments after arg buffer
  newKernelHeader.append(oldKernelHeader, structPos.back().second, oldKernelHeader.size() - structPos.back().second);

  // create local variables using argument buffer members
  for (const auto &s : structData)
  {
    newKernelHeader.append("\n");
    for (const auto& m : s.second)
    {
      string memberName = m.substr(m.rfind(' ')+1, m.size() - m.find(' '));
      if (useDefines)
      {
        newKernelHeader.append("#define "+memberName+" "+s.first+"_args->"+memberName+" \n");
      }
      else
      {
        newKernelHeader.append(m+" = "+s.first+"_args->"+memberName+"; \n");
      }
    }
  }

  return newKernelHeader;
}

inline string generateKernelEnding(const vector<pair<string, stringArr>>& structData,
                                   const uintPairList& structPos)
{
  string newKernelEnding;

  // create local variables using argument buffer members
  for (const auto &s : structData)
  {
    newKernelEnding.append("\n");
    for (const auto& m : s.second)
    {
      newKernelEnding.append("#undef "+m.substr(m.rfind(' ')+1, m.size() - m.find(' '))+"\n");
    }
  }

  return newKernelEnding;
}

string processArgumentBuffers(string source, bool useDefines = true)
{
  const string argBufferToken = "argumentStructMember";

  size_t kernelDecl = 0;
  do
  {
    // find the occurrence
    size_t pos = source.find(argBufferToken, kernelDecl);
    if (pos == source.npos)
    {
      break;
    }

    FunctionInfo kernelInfo = parseFunctionInfo(source, pos, false);
    kernelDecl              = kernelInfo.funcDeclOffset;
    size_t kernelDef        = kernelInfo.funcDefOffset;
    string kernelHeader     = kernelInfo.funcHeader;

    string newKernelHeader;
    vector<pair<string, stringArr>> structData;
    uintPairList structPos;

    size_t argOffset = 0;
    do
    {
      // find argument buffer token
      size_t tokenPos = kernelHeader.find(argBufferToken, argOffset);
      if (tokenPos == kernelHeader.npos)
      {
        break;
      }

      // argument buffer member declaration begin and end
      size_t argBegin = kernelHeader.find('(', tokenPos + argBufferToken.size());
      size_t argEnd   = kernelHeader.find(')', argBegin+1);

      // tokenize declaration
      string argDecl  = kernelHeader.substr(argBegin + 1, argEnd - argBegin - 1);
      stringArr tokens = tokenize(argDecl, ",");

      // add new structure if not already declared
      if (structData.empty() || structData.back().first != tokens[0])
      {
        structData.push_back(pair<string, stringArr>(tokens[0], stringArr()));
        structPos.push_back(uintPair(tokenPos, 0));
      }
      argOffset = argEnd+1;
      structPos.back().second = (uint)argOffset;
      structData.back().second.push_back(tokens[1]+" "+tokens[2]);
    }
    while (true);

    if (useDefines)
    {
      size_t end = findEndOfFunction(source, kernelDef + 1, '}');
      if (end != source.npos)
      {
        string ending = generateKernelEnding(structData, structPos);
        source.insert(end+1, ending);
      }
    }
    source.replace(kernelDecl, kernelDef - kernelDecl + 1, generateNewKernelHeader(kernelHeader, structData, structPos, useDefines));
  }
  while (true);

  return source;
}

string processAutoArgumentBuffers(string source, unordered_map<string, uintPairList>& kernelNameArgumentBufferMap)
{
  const string autoArgBufferToken = "#autoArgumentBuffer";

  // find the first occurrence
  size_t autoArgTokenPos = source.find(autoArgBufferToken);

  while (autoArgTokenPos != source.npos)
  {
    source.replace(autoArgTokenPos, autoArgBufferToken.size(), "");

    FunctionInfo kernelInfo = parseFunctionInfo(source, autoArgTokenPos, true);
    size_t kernelDecl       = kernelInfo.funcDeclOffset;
    size_t kernelDef        = kernelInfo.funcDefOffset;
    string kernelHeader     = kernelInfo.funcHeader;
    size_t argsBegin        = kernelInfo.argsBegin;
    size_t argsEnd          = kernelInfo.argsEnd;
    string kernelName       = kernelInfo.funcName;
    stringArr args          = kernelInfo.args;
    size_t kernelEnd        = kernelInfo.funcEndOffset;// findEndOfFunction(source, kernelDef + 1, '}');
    string kernelSource     = kernelInfo.funcSource;//source.substr(kernelDecl, kernelEnd - kernelDecl + 1);

    // find continuous constant arguments
    uint startIndex = -1;
    uintPairList argRanges;
    for (uint i=0; i<args.size(); i++)
    {
      const auto &arg = args[i];
      if (arg.find("const ") != arg.npos || arg.find("Const ") != arg.npos || arg.find("constantKernelInput") != arg.npos)
      {
        if (startIndex == -1)
        {
          startIndex = i;
        }
      }
      else
      {
        if (startIndex != -1)
        {
          argRanges.push_back(uintPair(startIndex, i-1));
          startIndex = -1;
        }
      }
    }

    if (startIndex != -1)
    {
      argRanges.push_back(uintPair(startIndex, (uint)args.size()-1));
    }

    // eleminate arguments with < 4 consecutive items
    for (uint i=0; i<argRanges.size(); i++)
    {
      if ((argRanges[i].second - argRanges[i].first) < 3)
      {
        argRanges.erase(argRanges.begin()+i);
        i--;
      }
    }

    kernelNameArgumentBufferMap[kernelName] = argRanges;

    vector<pair<string, stringArr>> structData;
    uintPairList structPos;

    for (const auto& range : argRanges)
    {
      for (uint i=range.first; i<=range.second; i++)
      {
        auto &arg = args[i];
        size_t pos = -1;
        stringArr parts;
        if (arg.find("constantKernelInput") != arg.npos)
        {
          pos = arg.find("constantKernelInput");
          arg.replace(pos, sizeof("constantKernelInput")-1, "");
          const size_t argsBegin = arg.find('(');
          parts = tokenize(arg.substr(argsBegin + 1, arg.size() - argsBegin - 1), ",");
        }
        else
        {
          pos = arg.find("const ");
          if (pos == arg.npos)
          {
            pos = arg.find("Const ");
          }
          const stringArr tokens = tokenize(arg.substr(pos, arg.size() - pos));
          parts.push_back("");
          for (uint t=0; t<tokens.size()-1; t++)
          {
            parts.back().append(tokens[t]+" ");
          }
          parts.push_back(tokens.back());
          parts[1] += ")";
        }
        args[i] = arg.substr(0, pos)+"argumentStructMember("+kernelName+"_"+to_string(range.first)+"_"+to_string(range.second)+", const "+parts[0]+", "+parts[1];
      }
    }

    string newHeader;
    for (const auto& s : args)
    {
      newHeader += s+",";
    }
    newHeader.pop_back();

//    size_t kernelEnd = findEndOfFunction(source, kernelDef + 1, '}');
//    string kernelSource = source.substr(kernelDecl, kernelEnd - kernelDecl + 1);
    source.replace(kernelDecl, kernelEnd - kernelDecl + 1, processArgumentBuffers(kernelSource.replace(argsBegin + 1, argsEnd - argsBegin - 1, newHeader), true));

    autoArgTokenPos = source.find(autoArgBufferToken, kernelDecl);
  }

  return source;
}

string addKernelInputs(const FunctionInfo& kernelInfo)
{
  string inputString("\n");

  for (int i=0; i<kernelInfo.args.size(); i++)
  {
    const string arg = trim(kernelInfo.args[i], " \t\r\n");
    if (arg == "KERNEL_GLOBAL_ARGUMENTS")
    {}
    else if (arg == "KERNEL_THREAD_ARGUMENTS")
    {
      inputString += "constantKernelInput(ushort3, threads_per_threadgroup);\n";
    }
    else if (arg == "KERNEL_THREADGROUP_ARGUMENTS")
    {
      inputString += "constantKernelInput(ushort3, threadgroups_per_grid);\n";
    }
    else
    {
      inputString += arg + ";\n";
    }
  }

  return inputString;
}

stringArr convertToHLSL(string source, bool keepInputArgs)
{
  stringArr kernels;
  size_t    kernelStart = 0;
  vector<size_t> kernelOffsets;

  do
  {
    FunctionInfo kernelInfo = parseFunctionInfo(source, kernelStart, true);
    size_t kernelDecl       = kernelInfo.funcDeclOffset;
    size_t kernelDef        = kernelInfo.funcDefOffset;
    string kernelHeader     = kernelInfo.funcHeader;
    size_t argsBegin        = kernelInfo.argsBegin;
    size_t argsEnd          = kernelInfo.argsEnd;
    string kernelName       = kernelInfo.funcName;
    stringArr args          = kernelInfo.args;
    size_t kernelEnd        = kernelInfo.funcEndOffset;
    string kernelSource     = kernelInfo.funcSource;

    if (kernelInfo.funcDeclOffset == source.npos) break;

    if (!keepInputArgs)
    {
      kernelSource.replace(argsBegin + 1, argsEnd - argsBegin - 1, "");
      kernelSource = addKernelInputs(kernelInfo) + kernelSource;
    }

    kernels.push_back(kernelSource);
    kernelOffsets.push_back(kernelDecl);

    source.replace(kernelDecl, kernelEnd - kernelDecl + 1, "");
    kernelStart = kernelDecl;
  }
  while (true);

  for (int i=0; i<kernels.size(); i++)
  {
    string kernelProgram = source;
    kernels[i] = kernelProgram.insert(kernelOffsets[i], kernels[i]);
  }

//  std::cout << kernels[0];

  return kernels;
}

vector<FunctionArgs> getKernelArgs(const string& source)
{
  vector<FunctionArgs> kernelInfos;
  size_t kernelStart = 0;
  vector<size_t> kernelOffsets;

  do
  {
    FunctionInfo funcInfo = parseFunctionInfo(source, kernelStart, true);
    size_t kernelDecl       = funcInfo.funcDeclOffset;
    size_t kernelDef        = funcInfo.funcDefOffset;
    string kernelHeader     = funcInfo.funcHeader;
    size_t argsBegin        = funcInfo.argsBegin;
    size_t argsEnd          = funcInfo.argsEnd;
    string kernelName       = funcInfo.funcName;
    stringArr args          = funcInfo.args;
    size_t kernelEnd        = funcInfo.funcEndOffset;
    string kernelSource     = funcInfo.funcSource;

    if (funcInfo.funcDeclOffset == source.npos) break;

    FunctionArgs info = funcInfo;

    for (auto& arg : info.args)
    {
      arg = trim(arg, " \t\r\n");
    }
    kernelInfos.push_back(info);

    kernelStart = kernelEnd;
  }
  while (true);

  return kernelInfos;
}
