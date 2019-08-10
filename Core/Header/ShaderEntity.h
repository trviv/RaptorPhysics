#ifndef SHADER_ENTITY_H
#define SHADER_ENTITY_H

#include "ComputeInterface.h"

/*
@class Class representing an shader using entities.
*/
class ShaderEntity
{

protected:

  vector<ComputeProgram>  programs;
  vector<ComputeKernel>   kernels;
  vector<string>          includeFiles;

  void registerShader(ComputeInterface* compute, const char* fileName, const vector<string>* oldType, const vector<string>* newType);
};

#endif
