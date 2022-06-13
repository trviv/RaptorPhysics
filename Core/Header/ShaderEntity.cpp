#include "ShaderEntity.h"

ShaderEntity::ShaderEntity()
{
  programs.clear();
  kernels.clear();
}

void ShaderEntity::registerShader(ComputeInterface* compute, const char* fileName, const vector<string>* oldType, const vector<string>* newType, const vector<string>* includeFiles)
{
  vector<string> localOld, localNew;

  if (oldType)
  {
    localOld.insert(localOld.begin(), oldType->begin(), oldType->end());
    localNew.insert(localNew.begin(), newType->begin(), newType->end());
  }

  localOld.push_back("ComputeSimdWidth");
  localNew.push_back(to_string(compute->simdSize()));

  localOld.push_back("ComputeSimdWidthExp");
  localNew.push_back(to_string(mCeilExpOf2(compute->simdSize())));

#ifdef USE_METAL_COMPUTE
  localOld.push_back("USE_METAL_COMPUTE");
  localNew.push_back("");
#endif

#ifdef USE_VULKAN_COMPUTE
  localOld.push_back("USE_METAL_COMPUTE");
  localNew.push_back("");

  localOld.push_back("USE_SIMD_COMPUTE");
  localNew.push_back("");
#endif

#ifdef USE_OPENCL_COMPUTE
  localOld.push_back("USE_OPENCL_COMPUTE");
  localNew.push_back("");
#endif

  programs.push_back(compute->createTemplateProgram(fileName, &localOld, &localNew, includeFiles));
}

void ShaderEntity::registerShaderFromSource(ComputeInterface* compute, const string& source, const vector<string>* oldType, const vector<string>* newType, const vector<string>* includeFiles)
{
  vector<string> localOld, localNew;

  if (oldType)
  {
    localOld.insert(localOld.begin(), oldType->begin(), oldType->end());
    localNew.insert(localNew.begin(), newType->begin(), newType->end());
  }

  localOld.push_back("ComputeSimdWidth");
  localNew.push_back(to_string(compute->simdSize()));

  localOld.push_back("ComputeSimdWidthExp");
  localNew.push_back(to_string(mCeilExpOf2(compute->simdSize())));

#ifdef USE_METAL_COMPUTE
  localOld.push_back("USE_METAL_COMPUTE");
  localNew.push_back("");
#endif

#ifdef USE_VULKAN_COMPUTE
  localOld.push_back("USE_METAL_COMPUTE");
  localNew.push_back("");

  localOld.push_back("USE_SIMD_COMPUTE");
  localNew.push_back("");
#endif

#ifdef USE_OPENCL_COMPUTE
  localOld.push_back("USE_OPENCL_COMPUTE");
  localNew.push_back("");
#endif

  programs.push_back(compute->createTemplateProgram(source, &localOld, &localNew, includeFiles));
}
