#include "PhysicsEntity.h"

PhysicsEntity::PhysicsEntity()
{
  constrainConstants.create(NULL, NULL, true);
  entityParticleSharedData.create(NULL, NULL, true);
  particleAuxData.create(NULL, NULL, true);
  particleRigidData.create(NULL, NULL, true);

  solver = SOLVER_NULL;

  ParticleSharedData sharedData;
  sharedData.velocityDamping = .99f;
  sharedData.sharedInvMass = 0.f;
  sharedData.sharedRadius = 0.f;
  sharedData.invMassIsShared = 0;
  entityParticleSharedData.host()->reserve(1);
  entityParticleSharedData.host()->push_back(sharedData);
}