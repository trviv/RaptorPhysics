/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef FLUID_SOLVER_PCISPH_SHADER
#define FLUID_SOLVER_PCISPH_SHADER

#include "FluidSolverPBF.shader"

/*
@kernel Predict position and velocity for the next step.
@param particlesNextPosition Next particle positions.
@param particlesNextVelocity Next particle velocities.
@param particlesPosition Integrated particle position.
@param nodeCount Total nodes in the solver.
*/
//#autoArgumentBuffer
Kernel void predictionStep(
  Device ParticleStruct*              particlesNextPosition,
  Device ParticleDifferential*        particlesNextVelocity,
  const Device ParticleStruct*        particlesPosition,
  const Device ParticleDifferential*  particlesDiff,
  const Device ParticleForce*         particlesPressureForce,
  const Device ParticleSharedData*    particleSharedData,
  constantKernelInput(uint,           nodeCount),
  constantKernelInput(float,          timeStep)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  // current particle data
  DECLARE_SELF_PARTICLE(particlesPosition, identity, nodeIdentity)

  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

  float3 velocity = particlesDiff[particleIndex].velocity + particlesPressureForce[particleIndex].force * timeStep * sharedData.sharedInvMass;
  particlesNextVelocity[particleIndex].velocity = velocity;

  selfParticle.position += velocity * timeStep;
  selfParticle.identity = identity;
  particlesNextPosition[particleIndex] = selfParticle;
}

/*
@kernel Get particle pressures.
@param particlesPressure Particle pressures output.
@param particlesDensity Particle densities.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param gridParticleCellIndex Computed cell index for each particle.
@param particlesPredictedOld Integrated particle position.
@param particleSharedData Particle entity shared data.
@param systemBoundingBox Physics system's bounding box.
@param invRadius Inverse of max particle radius in the system.
@param gridSize Size of grid in one dimension.
@param gridSizeExp Grid size in power of 2.
@param nodeCount Total nodes in the solver.
*/
//#autoArgumentBuffer
Kernel void calculatePressure(
  Device float*                     particlesPressure,
  const Device float*               particlesDensity,
  const Device uint*                gridCellParticleOffsets,
  const Device uint*                gridParticleCellIndex,
  const Device ParticleStruct*      particlesPredicted,
  const Device ParticleSharedData*  particleSharedData,
  Const XAB*                        systemBoundingBox,
  Const float*                      invRadius,
  constantKernelInput(ushort,       gridSize),
  constantKernelInput(ushort,       gridSizeExp),
  constantKernelInput(uint,         nodeCount),
  constantKernelInput(float,        timeStep),
  constantKernelInput(float,        invBeta)
  KERNEL_GLOBAL_ARGUMENTS)
{

  const uint particleIndex = threadIndex();

  if (particleIndex >= nodeCount)
  {
    return;
  }

  const uint gridCellIndex = gridParticleCellIndex[particleIndex];

  // current particle data
  float sumGradientMagnitude = 0.f;
  float3 sumGradientVector = constructFloat3(0.f);

  DECLARE_SELF_PARTICLE(particlesPredicted, identity, nodeIdentity)

  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  const float fluidKernelRadiusSq = sqr(sharedData.fluidSolverData.fluidKernelRadius);

  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_BEGIN
    const ParticleStruct otherParticle = particlesPredicted[otherNodeIndex];
    const float3 collisionVector = selfParticle.position - otherParticle.position;
    const float actualDistanceSq = lengthSq(collisionVector);

    if (actualDistanceSq >= fluidKernelRadiusSq)
    {
      continue;
    }

    const float actualDistance = sqrt(actualDistanceSq);
    const float3 gradient = collisionVector * spikyFunctionGradientVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius);
    sumGradientMagnitude += lengthSq(gradient);
    sumGradientVector += gradient;
  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END

  sumGradientVector *= sharedData.fluidSolverData.fluidKernelFunctionConstant[1];
  sumGradientMagnitude *= (sharedData.fluidSolverData.fluidKernelFunctionConstant[1] * sharedData.fluidSolverData.fluidKernelFunctionConstant[1]);

  const float invSigma = -(invBeta * sharedData.fluidSolverData.fluidKernelFunctionConstant[2]);
  particlesPressure[particleIndex] += -(particlesDensity[particleIndex] - 1.f/sharedData.invRestDensity) * invSigma;
}

/*
@kernel Calculate particle forces.
@param particlesPressureForce Force applied to the particle due to the fluid.
@param particlesDensity Particle densities output.
@param particlesPressure Particle pressures output.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param particlesPosition Integrated particle position.
@param particleDiff Particle velocity.
@param boundaryGridCellParticleOffsets Starting offset for each grid cell.
@param boundaryParticleCouplingData Calculated particle coupling data for boundary particles.
@param boundaryParticles Boundary particle position.
@param boundaryParticleDiff Boundary particle velocity.
@param particleSharedData Particle entity shared data.
@param gridParticleCellIndex Computed cell index for each particle.
@param nodeCount Total nodes in the solver.
@param occupiedCellCount Total active grid cells.
*/
//#autoArgumentBuffer
Kernel void calculateForcesPCISPH(
  Device ParticleForce*               particlesPressureForce,
  const Device float*                 particlesDensity,
  const Device float*                 particlesPressure,
  const Device uint*                  gridCellParticleOffsets,
  const Device uint*                  gridParticleCellIndex,
  const Device ParticleStruct*        particlesPosition,
  const Device ParticleDifferential*  particleDiff,
  const Device uint*                  boundaryGridCellParticleOffsets,
  const Device ParticleCouplingData*  boundaryParticleCouplingData,
  const Device ParticleStruct*        boundaryParticles,
  const Device ParticleDifferential*  boundaryParticleDiff,
  const Device ParticleSharedData*    particleSharedData,
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

  float3 force = constructFloat3(0.f);

  // current particle data
  DECLARE_SELF_PARTICLE(particlesPosition, identity, nodeIdentity)

  const ParticleSharedData sharedData = particleSharedData[nodeIdentity.entityId];
  const float selfDensity = particlesDensity[particleIndex];
  const float selfPressure = particlesPressure[particleIndex];

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  const float selfPressureByDensity = selfPressure/sqr(selfDensity);
  const float fluidKernelRadiusSq = sqr(sharedData.fluidSolverData.fluidKernelRadius);

  GRID_SOLVER_PACKED_NEIGHBOUR_PARTICLE_LOOP_BEGIN
    const ParticleStruct otherParticle = particlesPosition[otherNodeIndex];
    const float3 collisionVector = selfParticle.position - otherParticle.position;
    const float actualDistanceSq = lengthSq(collisionVector);

    if (actualDistanceSq >= fluidKernelRadiusSq)
    {
      continue;
    }

    const float actualDistance = sqrt(actualDistanceSq);
    const float density = selfPressureByDensity + particlesPressure[otherNodeIndex]/sqr(particlesDensity[otherNodeIndex]);
    force += collisionVector * (density * spikyFunctionGradientVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius));
  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END

  force = -(force * sharedData.fluidSolverData.fluidKernelFunctionConstant[1])/ sqr(sharedData.sharedInvMass);

  float3 boundaryForce = constructFloat3(0.f);
  const float3 selfParticleDiff = particleDiff[particleIndex].velocity;
  const float invDenominator = 0.5f/selfDensity;
  const float viscousFactor = sharedData.fluidSolverData.fluidKernelRadius * invDenominator;

  // calculate force contribution from boundary particles
  {
    // loop over boundary particles
    GRID_SOLVER_BOUNDARY_NEIGHBOUR_PARTICLE_LOOP_BEGIN
      const ParticleStruct otherParticle = boundaryParticles[otherNodeIndex];
      const float3 collisionVector = selfParticle.position - otherParticle.position;
      const float actualDistanceSq = lengthSq(collisionVector);

      if (actualDistanceSq >= fluidKernelRadiusSq)
      {
        continue;
      }

      const float3 velocityVector = selfParticleDiff - boundaryParticleDiff[otherNodeIndex].velocity;
      const float friction = -viscousFactor * min(dot(velocityVector, collisionVector), 0.f) / (actualDistanceSq + 0.01f * fluidKernelRadiusSq);

      const float actualDistance = sqrt(actualDistanceSq);

      // mass and rest density from this equation are multiplied to the end result
      boundaryForce += collisionVector * ((selfPressureByDensity + friction) * boundaryParticleCouplingData[otherNodeIndex].volume * spikyFunctionGradientVariable(actualDistance, sharedData.fluidSolverData.fluidKernelRadius));
    GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END

    boundaryForce = -boundaryForce * (sharedData.fluidSolverData.fluidKernelFunctionConstant[1] / (sharedData.sharedInvMass * sharedData.invRestDensity));
  }

  particlesPressureForce[particleIndex].force = force + boundaryForce;
}

/*
@kernel Calculate boundary particle forces.
@param boundaryParticlesForce Force applied to the particle due to the fluid.
@param particlesDensity Particle densities output.
@param particlesPressure Particle pressures output.
@param gridCellParticleOffsets Starting offset for each grid cell.
@param particlesPosition Integrated particle position.
@param particleDiff Particle velocity.
@param boundaryGridParticleCellIndex Computed cell index for boundary particles.
@param boundaryParticleCouplingData Calculated particle coupling data for boundary particles.
@param boundaryParticles Boundary particle position.
@param boundaryParticleDiff Boundary particle velocity.
@param boundaryGridParticleSystemIndex Bonundary particle's index to the system array.
@param particleSharedData Particle entity shared data.
@param gridParticleCellIndex Computed cell index for each particle.
@param nodeCount Total nodes in the solver.
@param occupiedCellCount Total active grid cells.
*/
//#autoArgumentBuffer
Kernel void calculateBoundaryForces(
  Device ParticleForce*               boundaryParticlesForce,
  const Device float*                 particlesDensity,
  const Device float*                 particlesPressure,
  const Device uint*                  gridCellParticleOffsets,
  const Device ParticleStruct*        particlesPosition,
  const Device ParticleDifferential*  particleDiff,
  const Device uint*                  boundaryGridParticleCellIndex,
  const Device ParticleCouplingData*  boundaryParticleCouplingData,
  const Device ParticleStruct*        boundaryParticles,
  const Device ParticleDifferential*  boundaryParticleDiff,
  const Device uint*                  boundaryGridParticleSystemIndex,
  const Device ParticleSharedData*    fluidParticleSharedData,
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

  const uint gridCellIndex = boundaryGridParticleCellIndex[particleIndex];

  float3 force = constructFloat3(0.f);

  // current particle data
  DECLARE_SELF_PARTICLE(boundaryParticles, identity, nodeIdentity)
  const float selfVolume = boundaryParticleCouplingData[particleIndex].volume;
  const float3 selfParticleDiff = boundaryParticleDiff[particleIndex].velocity;
  float selfPsiFluidMass; // derived from fluid

  // fluid particle data
  FluidSolverData fluidSolverData;
  IdentityInfo fluidIdentity;
  float viscousFactor;

  fluidIdentity.identity = -1;

#ifdef GRID_SOLVER_HASH_FUNCTION
  const float3 particleCellPosition = (selfParticle.position - systemBoundingBox->min) * invRadius[0];
#endif

  float fluidKernelRadiusSq;

  GRID_SOLVER_PACKED_NEIGHBOUR_PARTICLE_LOOP_BEGIN
    const ParticleStruct otherParticle = particlesPosition[otherNodeIndex];

    // if current fluid properties are stored, store it and update derived variables
    if (fluidIdentity.identity != otherParticle.identity.identity)
    {
      const ParticleNodeIdentity nodeIdentity = uncompressToNodeIdentity(otherParticle.identity);
      ParticleSharedData fluidSharedData = fluidParticleSharedData[nodeIdentity.entityId];
      selfPsiFluidMass = selfVolume / (fluidSharedData.invRestDensity * fluidSharedData.sharedInvMass);
      fluidSolverData = fluidSharedData.fluidSolverData;
      fluidKernelRadiusSq = sqr(fluidSharedData.fluidSolverData.fluidKernelRadius);
      fluidIdentity.identity = otherParticle.identity.identity;
      viscousFactor = 0.5f * fluidSolverData.fluidKernelRadius;
    }

    const float3 collisionVector = selfParticle.position - otherParticle.position;
    const float actualDistanceSq = lengthSq(collisionVector);

    if (actualDistanceSq >= fluidKernelRadiusSq)
    {
      continue;
    }

    const float otherParticleDensityInv = 1.f/particlesDensity[otherNodeIndex];

    const float3 velocityVector = selfParticleDiff - particleDiff[otherNodeIndex].velocity;
    const float friction = -viscousFactor * otherParticleDensityInv * min(dot(velocityVector, collisionVector), 0.f) / (actualDistanceSq + 0.01f * fluidKernelRadiusSq);
    const float actualDistance = sqrt(actualDistanceSq);
    const float otherPressureByDensity = particlesPressure[otherNodeIndex] * sqr(otherParticleDensityInv);

    force += collisionVector * (selfPsiFluidMass * (otherPressureByDensity + friction) * fluidSolverData.fluidKernelFunctionConstant[1] * spikyFunctionGradientVariable(actualDistance, fluidSolverData.fluidKernelRadius));
  GRID_SOLVER_NEIGHBOUR_PARTICLE_LOOP_END

  boundaryParticlesForce[boundaryGridParticleSystemIndex[particleIndex]].force -= force;
}

#endif
