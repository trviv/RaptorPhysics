#ifndef FLUID_SOLVER_COMMON_SHADER
#define FLUID_SOLVER_COMMON_SHADER

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

//#ifdef COMPUTE_SHADER_SCOPE

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

#ifdef COMPUTE_SHADER_SCOPE

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
  uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  particleIndex = gridCellParticleIndices[threadIndex()];

  gridParticleCellIndexNew[threadIndex()] = gridParticleCellIndexOld[particleIndex];
  particlesNew[threadIndex()] = particlesOld[particleIndex];
  particlesPredictedNew[threadIndex()] = particlesPredictedOld[particleIndex];
  particleDiffNew[threadIndex()] = particleDiffOld[particleIndex];
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
