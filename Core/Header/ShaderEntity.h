/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef SHADER_ENTITY_H
#define SHADER_ENTITY_H

#include <Compute/ComputeInterface.h>

/*!
@class Class representing an shader using entities.
*/
class ShaderEntity
{
protected:

  vector<ComputeProgram>  programs;
  vector<ComputeKernel>   kernels;

  ShaderEntity();

  void registerShader(ComputeInterface* compute, const char* fileName, const vector<string>* oldType, const vector<string>* newType, const vector<string>* includeFiles = NULL);

  void registerShaderFromSource(ComputeInterface* compute, const string& source, const vector<string>* oldType, const vector<string>* newType, const vector<string>* includeFiles = NULL);
};

#endif
