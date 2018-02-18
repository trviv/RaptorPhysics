#include "PhysicsEntity.h"

PhysicsEntity::PhysicsEntity()
{
  constrainConstants.create(NULL, NULL, true);
  particleSharedData.create(NULL, NULL, true);
  particleAuxData.create(NULL, NULL, true);
  particleRigidData.create(NULL, NULL, true);

  instanceCount = 1;
  solver = 0;

  ParticleSharedData sharedData;
  sharedData.velocityDamping = .99f;
  sharedData.sharedInvMass = 0.f;
  sharedData.sharedRadius = 0.f;
  sharedData.invMassIsShared = 0;
  particleSharedData.host()->push_back(sharedData);
}