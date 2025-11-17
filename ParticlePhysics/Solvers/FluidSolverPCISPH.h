/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef FLUID_SOLVER_PCISPH_H
#define FLUID_SOLVER_PCISPH_H

#include "FluidSolver.h"

/*!
@class Class to solve fluid constraints using Predictive-Corrective Incompressible SPH.
*/
class FluidSolverPCISPH : public FluidSolver
{
protected:

  float invBeta;

  DeviceArray <float> &particlesPressure = particlesLambda;
  DeviceArray <ParticleStruct> &particlesNextPosition   = particlesTemp[1];
  DeviceArray <ParticleStruct> &particlesNextVelocity   = particlesBufferTemp;

  void update();

public:

  FluidSolverPCISPH(ComputeInterface* compute, SharedAllocator* allocator);

  void create(ComputeInterface* compute);

  void solve(float timeStep);
};

#endif
