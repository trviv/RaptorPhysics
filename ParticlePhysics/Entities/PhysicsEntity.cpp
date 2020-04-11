#include "PhysicsEntity.h"

PhysicsEntity::PhysicsEntity()
{
  constrainConstants.create(NULL, NULL, true);
  entitySharedData.create(NULL, NULL, true);
  particleAuxData.create(NULL, NULL, true);
  particleRigidData.create(NULL, NULL, true);
  particleCollisionData.create(NULL, NULL, true);

  solver = SOLVER_NULL;

  ParticleSharedData sharedData;
  sharedData.collisionSolverData.velocityDamping = .995f;
  sharedData.collisionSolverData.collisionDamping = .98f;
  setInvMassIsShared(sharedData, true);
  sharedData.sharedInvMass = 0.f;
  setRadiusIsShared(sharedData, true);
  sharedData.sharedRadius = 0.f;
  sharedData.viscosity = 0.0001f;
  sharedData.collisionSolverData.kineticFrictionCoef = 0.9f;
  sharedData.collisionSolverData.staticFrictionCoef = 0.9f;
  sharedData.fluidSolverData.fluidKernelRadius = 1.f;
  sharedData.invRestDensity = 1.f/997.f;
  sharedData.gasConstantK = 0.001f;
  setCollisionDataIsShared(sharedData, true);
  entitySharedData.host()->reserve(1);
  entitySharedData.host()->push_back(sharedData);
}

PhysicsEntity::~PhysicsEntity()
{
}
