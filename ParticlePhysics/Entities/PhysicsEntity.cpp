#include "PhysicsEntity.h"

PhysicsEntity::PhysicsEntity()
{
  constrainConstants.create(NULL, NULL, true);
  particleSharedData.create(NULL, NULL, true);
  particleAuxData.create(NULL, NULL, true);
  particleRigidData.create(NULL, NULL, true);

  identity = -1;
  solver = SOLVER_NULL;

  ParticleSharedData sharedData;
  sharedData.velocityDamping = .99f;
  sharedData.sharedInvMass = 0.f;
  sharedData.sharedRadius = 0.f;
  sharedData.invMassIsShared = 0;
  particleSharedData.host()->push_back(sharedData);

  for (uint i = 0; i < SECTION_DATA_MAX; i++)
  {
    sectionShared[i] = false;
  }
}