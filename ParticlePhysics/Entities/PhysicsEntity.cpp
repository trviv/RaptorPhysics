/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "PhysicsEntity.h"

PhysicsEntity::PhysicsEntity()
{
  constrainConstants.create(NULL, NULL);
  entitySharedData.create(NULL, NULL);
  particleRigidData.create(NULL, NULL);
  particleCollisionData.create(NULL, NULL);
  particleCouplingData.create(NULL, NULL);

  solver = SOLVER_NULL;

  ParticleSharedData sharedData;
  sharedData.collisionSolverData.velocityDamping = 0.995f;
  sharedData.collisionSolverData.collisionDamping = 0.995f;
  setInvMassIsShared(sharedData, true);
  sharedData.sharedInvMass = 0.f;
  setRadiusIsShared(sharedData, true);
  sharedData.sharedRadius = 0.f;
  sharedData.viscosity = 0.0001f;
  sharedData.collisionSolverData.kineticFrictionCoef = 0.9f;
  sharedData.collisionSolverData.staticFrictionCoef = 0.9f;
  sharedData.fluidSolverData.fluidKernelRadius = 1.f;
  sharedData.invRestDensity = 1.f/997.f;
  sharedData.gasConstantK = 0.1f;
  setCollisionDataIsShared(sharedData, true);
  entitySharedData.host()->reserve(1);
  entitySharedData.host()->push_back(sharedData);
}

PhysicsEntity::~PhysicsEntity()
{
}
