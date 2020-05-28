#ifndef FLUID_SOLVER_COMMON_SHADER
#define FLUID_SOLVER_COMMON_SHADER

//#define FLUID_SOLVER_SORTED_REARRANGE

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

void bitonicSortSharedUint3(
  Shared uint3* localNodes,
  const short   maxDepth,
  const short   localThreadCount,
  const short   localIndex)
{
  localMemBarrier();

  for (short i=0; i<maxDepth; i++)
  {
    for (short j=i; j>=0; j--)
    {
      const short stride = (1 << j);
      const short index1 = localIndex << (j + 1);
      const short index2 = index1 + stride;

      uint3 node1;
      uint3 node2;

      if (index2 < localThreadCount)
      {
        node1 = localNodes[index1];
        node2 = localNodes[index2];
      }
      localMemBarrier();

      if (index2 < localThreadCount)
      {
        if (node1.x > node2.x && node1.y == node2.y)
        {
          localNodes[index1] = node2;
          localNodes[index2] = node1;
        }
      }
      localMemBarrier();
    }
  }
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
#ifdef FLUID_SOLVER_SORTED_REARRANGE
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(uint,           nodeCount),
  constantKernelInput(int,            gridSize),
  constantKernelInput(int,            gridSizeExp),
  sharedMemKernelInput(uint3,         particleSpatialData,  14)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const short elements = 2;
  const short scale = 8;

  for (short i=0; i<elements; i++)
  {
    particleSpatialData[threadLocalIndex() + i * threadGroupSize()] = constructUint3(-1);
  }

  for (short i=0; i<elements; i++)
  {
    const uint threadGlobalIndex = threadLocalIndex() + threadGroupSize() * (i + threadGroupIndex() * elements);
    if (threadGlobalIndex < nodeCount)
    {
      const uint particleIndex = gridCellParticleIndices[threadGlobalIndex];
      const uint gridCellIndex = gridParticleCellIndexOld[particleIndex];
      const ParticleStruct selfParticle = particlesPredictedOld[particleIndex];
      const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
//      const uint cellInternalSpatialIndex = encodeGridIndexInt3(constructInt3((invRadius[0] * selfParticle.position - floor(particleCellPosition)) * scale), scale);
      const uint cellInternalSpatialIndex = get32BitMortonCode(constructInt3((invRadius[0] * selfParticle.position - floor(particleCellPosition)) * scale));

      particleSpatialData[threadLocalIndex() + i * threadGroupSize()] = constructUint3(cellInternalSpatialIndex, gridCellIndex, particleIndex);
    }
  }

  const short tgSizePowOf2 = sizeof(threadGroupSize()) * 8 - clz(threadGroupSize()) - 1;

  bitonicSortSharedUint3(particleSpatialData, tgSizePowOf2, threadGroupSize()*elements, threadLocalIndex());

  for (short i=0; i<elements; i++)
  {
    const uint2 particleData = particleSpatialData[threadLocalIndex() + i * threadGroupSize()].yz;
    const uint threadGlobalIndex = threadLocalIndex() + threadGroupSize() * (i + threadGroupIndex() * elements);

    if (particleData.y != -1)
    {
      gridParticleCellIndexNew[threadGlobalIndex] = particleData.x;
      particlesNew[threadGlobalIndex] = particlesOld[particleData.y];
      particlesPredictedNew[threadGlobalIndex] = particlesPredictedOld[particleData.y];
      particleDiffNew[threadGlobalIndex] = particleDiffOld[particleData.y];
    }
  }
#else
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
#endif
}

#define GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_BEGIN \
  GRID_SOLVER_NEIGHBOUR_LOOP_BEGIN \
    const uint2 indexRange = getRangeFromOffset(gridCellParticleOffsets, gridCellIndex); \
    /* iterate over particles in neighboring cells*/ \
    for (int otherNodeIndex = indexRange.x; otherNodeIndex < indexRange.y; otherNodeIndex++) \
    {

#define GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END \
    } \
  GRID_SOLVER_NEIGHBOUR_LOOP_END

inline float calculateParticleDensity(
  const ParticleStruct          selfParticle,
  const float                   selfMass,
  const FluidSolverData         fluidSolverData,
  const Device uint*            gridCellParticleOffsets,
  const Device ParticleStruct*  particles,
  const uint                    gridCellIndex,
  const short                   gridSize,
  const short                   gridSizeExp)
{
  float density = 0.f;
  const float fluidKernelRadiusSq = sqr(fluidSolverData.fluidKernelRadius);

  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_BEGIN
    // iterate over each particle in the cell
    const float3 collisionVector = selfParticle.position - particles[otherNodeIndex].position;
    const float actualDistanceSq = lengthSq(collisionVector);

    density += select(0.f, poly6FunctionVariableSquares(actualDistanceSq, fluidKernelRadiusSq), actualDistanceSq < fluidKernelRadiusSq);
  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END

  return density * fluidSolverData.fluidKernelFunctionConstant[0] * selfMass;
}

/*
@kernel Calculate fluid particle density.
@param particlesDensity Calculated particle density.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridParticleCellIndex Computed cell index for each particle.
@param particlesPredicted Integrated particle position.
@param particleSharedData Particle entity shared data.
@param systemBoundingBox Physics system's bounding box.
@param invRadius Inverse of max particle radius in the system.
@param gridSize Size of grid in one dimension.
@param gridSizeExp Grid size in power of 2.
@param nodeCount Total nodes in the solver.
*/
Kernel void calculateDensity(
  Device float*                       particlesDensity,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleStruct*        particlesPredicted,
  const Device ParticleSharedData*    particleSharedData,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(int,            gridSize),
  constantKernelInput(int,            gridSizeExp),
  constantKernelInput(uint,           nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];
  DECLARE_SELF_PARTICLE(particlesPredicted, identity, nodeIdentity)
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
  const float mass = 1.f/particleSharedData[nodeIdentity.entityId].sharedInvMass;
  const FluidSolverData fluidSolverData = particleSharedData[nodeIdentity.entityId].fluidSolverData;

  particlesDensity[particleIndex] = calculateParticleDensity(selfParticle, mass, fluidSolverData, gridCellParticleOffsets, particlesPredicted, gridCellIndex, gridSize, gridSizeExp);
}

inline float3 surfaceTensionAkinci(const float3 collisionVector, const float actualDistance, const float selfParticleMass, const float otherParticleMass, const ParticleSharedData sharedData)
{
  float h = sharedData.fluidSolverData.fluidKernelRadius;
  float h_3 = h * h * h;
  float h_6 = h_3 * h_3;
  float h_9 = h_6 * h_3;
  float x = (h - actualDistance);
  x = x * x * x;
  float y = x * actualDistance * actualDistance * actualDistance;
  return collisionVector * (selfParticleMass * otherParticleMass * sharedData.surfaceTensionCoeff * (32.f / (3.1415926535897932384626433832795 * h_9 * actualDistance)) * select(2.f * y - h_6 / 64.f, y, (2.f * actualDistance) > h));
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
