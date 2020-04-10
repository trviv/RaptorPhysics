#ifndef FLUID_SOLVER_PCISPH_H
#define FLUID_SOLVER_PCISPH_H

#include "FluidSolver.h"

/*!
@class Class to solve fluid constraints using Predictive-Corrective Incompressible SPH.
*/
class FluidSolverPCISPH : public FluidSolver
{
protected:

  float beta;

  DeviceArray <float> &particlesPressure = particlesLambda;
  DeviceArray <ParticleStruct> &particlesPressureForce  = particlesTemp[0];
  DeviceArray <ParticleStruct> &particlesNextPosition   = particlesTemp[1];
  DeviceArray <ParticleStruct> &particlesNextVelocity   = particlesBufferTemp;

public:

  FluidSolverPCISPH(ComputeInterface* compute, SharedAllocator* allocator);

  void create(ComputeInterface* compute);

  void solve(float timeStep);
};

#endif
