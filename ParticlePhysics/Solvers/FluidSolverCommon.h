#ifndef FLUID_SOLVER_COMMON_SHADER
#define FLUID_SOLVER_COMMON_SHADER

#ifndef COMPUTE_SHADER_SCOPE

inline float poly6FunctionConstant(const float h)
{
  const float x = 1.f / (h * h * h);
  return (315.f / (64.f * M_PI_F)) * x * x * x;
}

inline float spikyFunctionConstant(const float h)
{
  const float x = 1.f / (h * h);
  return (15.f / M_PI_F) * x * x * x;
}

inline float viscosityFunctionConstant(const float h)
{
  const float x = 1.f / (h * h);
  return (45.f / M_PI_F) * x * x * x;
}

#else

inline float poly6FunctionVariable(const float r, const float h)
{
  const float x = (h * h - r * r);
  return x * x * x;
}

inline float poly6FunctionVariableSquares(const float rSquared, const float hSquared)
{
  const float x = (hSquared - rSquared);
  return x * x * x;
}

inline float poly6FunctionGradientVariable(const float r, const float h)
{
  const float x = (h * h - r * r);
  return -6.f * x * x;
}

inline float poly6FunctionGradientVariableSquares(const float rSquared, const float hSquared)
{
  const float x = (hSquared - rSquared);
  return -6.f * x * x;
}

inline float poly6FunctionLaplacianVariable(const float r, const float h)
{
  const float x = (h * h - r * r) * (3 * h * h - 7 * r * r);
  return -6.f * x;
}

inline float poly6FunctionLaplacianVariableSquares(const float rSquared, const float hSquared)
{
  const float x = (hSquared - rSquared) * (3 * hSquared - 7 * rSquared);
  return -6.f * x;
}

inline float spikyFunctionVariable(const float r, const float h)
{
  const float x = (h - r);
  return x * x * x;
}

inline float spikyFunctionGradientVariable(const float r, const float h)
{
  const float x = (h - r);
  return -3.f * x * x;
}

inline float viscosityFunctionLaplacianVariable(const float r, const float h)
{
  const float x = (h - r);
  return x;
}

/*
@kernel Kernel to reorder particles in buffers based on gird index.
@param gridCellParticleIndices Output array for particle indices.
@param nodeCount Total nodes in the solver.
*/
Kernel void reorderFluidParticles(
  Device ParticleStruct*              particlesNew,
  const Device ParticleStruct*        particlesOld,
  Device ParticleDifferential*        particleDiffNew,
  const Device ParticleDifferential*  particleDiffOld,
  Device ParticleStruct*              particlesPredictedNew,
  const Device ParticleStruct*        particlesPredictedOld,
  Device uint*                        gridParticleCellIndexNew,
  const Device uint*                  gridParticleCellIndexOld,
  const Device uint*                  gridCellParticleIndices,
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  uint particleOffset = threadIndex() * 4;

  if (particleOffset >= nodeCount)
  {
    return;
  }

  uint particleCount = min(nodeCount, particleOffset+4) - particleOffset;

  uint particleIndices[4];
  *((Thread uint4*)particleIndices) = ((const Device uint4*)gridCellParticleIndices)[threadIndex()];

  uint gridParticleCellIndexTemp[4];

  for (int i=0; i<particleCount; i++)
  {
    gridParticleCellIndexTemp[i] = gridParticleCellIndexOld[particleIndices[i]];
  }

  if (particleCount == 4)
  {
    ((Device uint4*)gridParticleCellIndexNew)[threadIndex()] = *((Thread uint4*)gridParticleCellIndexTemp);
  }
  else
  {
    for (int i=0; i<particleCount; i++)
    {
      gridParticleCellIndexNew[particleOffset + i] = gridParticleCellIndexTemp[i];
    }
  }

  ParticleStruct particlesTemp[4];

  for (int i=0; i<particleCount; i++)
  {
    particlesTemp[i] = particlesOld[particleIndices[i]];
  }

  if (particleCount == 4)
  {
    ((Device commonUint16*)particlesNew)[threadIndex()] = *((Thread commonUint16*)particlesTemp);
  }
  else
  {
    for (int i=0; i<particleCount; i++)
    {
      particlesNew[particleOffset + i] = particlesTemp[i];
    }
  }

  for (int i=0; i<particleCount; i++)
  {
    particlesTemp[i] = particlesPredictedOld[particleIndices[i]];
  }

  if (particleCount == 4)
  {
    ((Device commonUint16*)particlesPredictedNew)[threadIndex()] = *((Thread commonUint16*)particlesTemp);
  }
  else
  {
    for (int i=0; i<particleCount; i++)
    {
      particlesPredictedNew[particleOffset + i] = particlesTemp[i];
    }
  }

  ParticleDifferential particlesDiffTemp[4];

  for (int i=0; i<particleCount; i++)
  {
    particlesDiffTemp[i] = particleDiffOld[particleIndices[i]];
  }

  if (particleCount == 4)
  {
    ((Device commonUint16*)particleDiffNew)[threadIndex()] = *((Thread commonUint16*)particlesDiffTemp);
  }
  else
  {
    for (int i=0; i<particleCount; i++)
    {
      particleDiffNew[particleOffset + i] = particlesDiffTemp[i];
    }
  }
}

/*inline float poly6Function(const float r, const float h)
{
  const float x = (h * h - r * r) / (h * h * h);
  return (315.f / (64.f * M_PI_F)) * x * x * x;
}

inline float poly6FunctionGradient(const float r, const float h)
{
  const float invH = 1.f/h;
  const float x = (h * h - r * r) * sqr(sqr(invH * invH));
  return (-945.f / (32.f * M_PI_F)) * x * x * invH;
}

inline float poly6FunctionLaplacian(const float r, const float h)
{
  const float x = (h * h - r * r) * (-7 * r * r + 3 * h * h) / (h * sqr(sqr(h * h)));
  return (945.f / (32.f * M_PI_F)) * x;
}

inline float spikyFunction(const float r, const float h)
{
  const float x = (h - r) / (h * h);
  return (15.f / M_PI_F) * x * x * x;
}

inline float spikyFunctionGradient(const float r, const float h)
{
  const float x = (h - r) / (h * h * h);
  return -(45.f / M_PI_F) * x * x;
}

inline float viscosityFunctionLaplacian(const float r, const float h)
{
  const float x = (h - r) / sqr(h * h * h);
  return (45.f / M_PI_F) * x;
}*/

#endif

#endif
