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
  sharedData.velocityDamping = .98f;
  sharedData.invMassIsShared = 0;
  sharedData.sharedInvMass = 0.f;
  sharedData.radiusIsShared = 1;
  sharedData.sharedRadius = 0.f;
  sharedData.collisionDataIsShared = 1;
  sharedData.sharedCollisionData.sdfGradient = float3(0, 0, 0);
  sharedData.sharedCollisionData.sdfGradient2 = float3(0, 0, 0);
  entitySharedData.host()->reserve(1);
  entitySharedData.host()->push_back(sharedData);
}