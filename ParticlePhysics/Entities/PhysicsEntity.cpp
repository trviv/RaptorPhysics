#include "PhysicsEntity.h"

PhysicsEntity::PhysicsEntity()
{
  constrainConstants.create(NULL, NULL, true);
  entitySharedData.create(NULL, NULL, true);
  particleAuxData.create(NULL, NULL, true);
  particleRigidData.create(NULL, NULL, true);

  solver = SOLVER_NULL;

  ParticleSharedData sharedData;
  sharedData.velocityDamping = .99f;
  sharedData.sharedInvMass = 0.f;
  sharedData.sharedRadius = 0.f;
  sharedData.invMassIsShared = 0;
  entitySharedData.host()->reserve(1);
  entitySharedData.host()->push_back(sharedData);
}