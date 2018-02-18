#include "ShaderEntity.h"

void ShaderEntity::registerShader(ComputeInterface* compute, const char* fileName, const vector<string>* oldType, const vector<string>* newType)
{
  programs.push_back(compute->createTemplateProgram(fileName, oldType, newType, &includeFiles));
}