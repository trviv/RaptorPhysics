#include "ShaderEntity.h"
#include "Math.h"

void ShaderEntity::registerShader(ComputeInterface* compute, const char* fileName, const vector<string>* oldType, const vector<string>* newType)
{
  vector<string> localOld, localNew;

  if (oldType)
  {
    localOld.insert(localOld.begin(), oldType->begin(), oldType->end());
    localNew.insert(localNew.begin(), newType->begin(), newType->end());
  }

  localOld.push_back("COMPUTE_SUB_GROUP_SIZE");
  localNew.push_back(to_string(compute->simdSize()));

  localOld.push_back("COMPUTE_SUB_GROUP_EXP");
  localNew.push_back(to_string(mCeilExpOf2(compute->simdSize())));

  programs.push_back(compute->createTemplateProgram(fileName, &localOld, &localNew, &includeFiles));
}