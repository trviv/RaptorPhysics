#ifndef FLUID_SOLVER_SHADER
#define FLUID_SOLVER_SHADER

#define FLUID_SOLVER_COLOR_GRADIENT_THRESHOLD 0.1f

/*
@kernel Resolve particle collisions.
@param particleForce Force applied to the particle due to the fluid.
@param particlesDensity Particles density.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridParticleCellIndex Computed cell index for each particle.
@param particlesPredicted Integrated particle position.
@param particleSharedData Particle entity shared data.
@param nodeCount Total nodes in the solver.
*/
//#autoArgumentBuffer
Kernel void calculateForces(
  Device ParticleForce*               particleForce,
  const Device float*                 particlesDensity,
  const Device ParticleDifferential*  particleDiff,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleStruct*        particlesPredicted,
  const Device ParticleSharedData*    particleSharedData,
  Device ParticleCollisionData*       particleCollisionData,
  Const XAB*                          systemBoundingBox,
  Const float*                        invRadius,
  constantKernelInput(ushort,         gridSize),
  constantKernelInput(ushort,         gridSizeExp),
  constantKernelInput(uint,           nodeCount),
  constantKernelInput(float,          timeStep)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  float3 pressureForce = constructFloat3(0.f);
  float3 viscosityForce = constructFloat3(0.f);
  float3 colorGradient = constructFloat3(0.f);
  float3 colorLaplacian = constructFloat3(0.f);

  // current particle data
  DECLARE_SELF_PARTICLE(particlesPredicted, identity, nodeIdentity)

  const float3 selfParticleDiff = particleDiff[particleIndex].velocity;
  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
  const float density = particlesDensity[particleIndex];
  const float restDensity = 1.f/sharedData.invRestDensity;
  const float selfPressureTerm = sharedData.gasConstantK * (density - restDensity);

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  const float fluidKernelRadiusSq = sqr(sharedData.fluidSolverData.fluidKernelRadius);

  GRID_SOLVER_PACKED_NEIGHBOUR_PARTICLE_LOOP_BEGIN
    const ParticleStruct otherParticle = particlesPredicted[otherNodeIndex];

    const float3 collisionVector = selfParticle.position - otherParticle.position;
    const float actualDistanceSq = lengthSq(collisionVector);

    if (actualDistanceSq >= fluidKernelRadiusSq)
    {
      continue;
    }

    const float actualDistance = sqrt(actualDistanceSq);
    const float currentParticleDensity = particlesDensity[otherNodeIndex];
    const float currentParticleDensityInv = 1.f/currentParticleDensity;

    // force due to viscosity
    const float3 velocityVector = particleDiff[otherNodeIndex].velocity - selfParticleDiff;
    const float viscosityTerm = viscosityFunctionLaplacianVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius);

    viscosityForce += velocityVector * (viscosityTerm * currentParticleDensityInv);

    // force due to pressure
    const float pressureTerm = (selfPressureTerm + sharedData.gasConstantK * (density - restDensity)) * currentParticleDensityInv * 0.5f;

    pressureForce -= collisionVector * (pressureTerm * spikyFunctionGradientVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius));
    colorGradient += collisionVector * poly6FunctionGradientVariableSquares(actualDistanceSq, fluidKernelRadiusSq) * currentParticleDensityInv;
    colorLaplacian += collisionVector * poly6FunctionLaplacianVariableSquares(actualDistanceSq, fluidKernelRadiusSq) * currentParticleDensityInv;
  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END


  colorGradient *= sharedData.fluidSolverData.fluidKernelFunctionConstant[0];
  colorLaplacian *= sharedData.fluidSolverData.fluidKernelFunctionConstant[0];
  pressureForce *= sharedData.fluidSolverData.fluidKernelFunctionConstant[1];
  viscosityForce *= sharedData.viscosity * sharedData.fluidSolverData.fluidKernelFunctionConstant[2];

  float3 color = colorLaplacian;
  const float colorGradientLength = length(colorGradient);
  colorGradient /= colorGradientLength;
  if (colorGradientLength >= FLUID_SOLVER_COLOR_GRADIENT_THRESHOLD)
  {
    color *= -0.1f * colorGradient;
  }

  particleCollisionData[particleIndex].gradientMagnitude = colorGradientLength * timeStep;
  particleCollisionData[particleIndex].transformedSdfGradient = encodeDirection(select(-colorGradient, constructFloat3(0.f), colorGradientLength <= FLUID_SOLVER_COLOR_GRADIENT_THRESHOLD));

  particleForce[particleIndex].force = (pressureForce + viscosityForce + color) / sharedData.sharedInvMass;
}

#endif
