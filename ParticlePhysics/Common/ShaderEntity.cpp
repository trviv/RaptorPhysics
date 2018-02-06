#include "ShaderEntity.h"

void ShaderEntity::registerShader(ComputeInterface* compute, const char* fileName, const vector<string>* oldType, const vector<string>* newType)
{
  const vector<string> includeFiles = { "ComputeHeader.shader", "ConstrainStruct.h", "ParticleStruct.h", "EntityHeader.h" };
  programs.push_back(compute->createTemplateProgram(fileName, oldType, newType, &includeFiles));
}