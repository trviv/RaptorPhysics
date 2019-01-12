#ifndef PARTICLE_STRUCT_H
#define PARTICLE_STRUCT_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Core.h>
typedef Real3 float3;
typedef Real3 float4;
#endif

#define PHYSICS_ENTITY_ID_MASK    0xFFFFF
#define PHYSICS_INSTANCE_ID_SHIFT 20


struct IdentityInfo_t
{
  uint identity[2];

#ifndef COMPUTE_SHADER_SCOPE

  IdentityInfo_t()
  {
    identity[0] = -1;
    identity[1] = -1;
  }

  void setEntityId(uint solver, uint entityId)
  {
    identity[0] = (mCeilExpOf2(solver) << PHYSICS_INSTANCE_ID_SHIFT) | (entityId & PHYSICS_ENTITY_ID_MASK);
  }

  void setInstanceId(uint instanceId)
  {
    identity[1] = instanceId;
  }

#endif
};

typedef struct IdentityInfo_t IdentityInfo;
typedef struct IdentityInfo_t PhysicsEntityId;

static uint getInstanceId(const IdentityInfo particleIdentity)
{
  return particleIdentity.identity[1];
}

static uint getEntityId(const IdentityInfo particleIdentity)
{
  return particleIdentity.identity[0] & PHYSICS_ENTITY_ID_MASK;
}

static uint getSolverType(const IdentityInfo particleIdentity)
{
  return (particleIdentity.identity[0] >> PHYSICS_INSTANCE_ID_SHIFT) + 1;
}


/*
@struct Allocation data shared by all the particles of an entity.
*/
struct EntityLocation_t
{
  PartitionInfo node;
  PartitionInfo connection;
};

typedef struct EntityLocation_t EntityLocation;


/*
@struct Allocation data shared by all the particles of an entity.
*/
struct DEFAULT_ALIGN GroupData_t
{
  uint    minIdentity;
  uint    maxIdentity;
};

typedef struct GroupData_t GroupData;


struct DEFAULT_ALIGN ParticleCollisionData_t
{
  union
  {
    struct
    {
      float3  sdfGradient;
    };
    struct
    {
      float dummy[3];
      float sdfMagnitude;
    };
  };
  float3  sdfGradient2;
};

typedef struct ParticleCollisionData_t ParticleCollisionData;


/*
@struct Data shared by all the particles of an entity.
*/
struct DEFAULT_ALIGN ParticleSharedData_t
{
  /*@member If mass is shared by particles of a body.*/
  uint  invMassIsShared;
  /*@member Shared inverse mass.*/
  float sharedInvMass;

  /*@member If radius is shared by particles of a body.*/
  uint  radiusIsShared;
  /*@member Shared radius.*/
  float sharedRadius;

  /*@member Stiffness for spring constraint.*/
  float stiffness;
  /*@member Viscosity for fluid constraint.*/
  float viscosity;
  /*@member Velocity damping.*/
  float velocityDamping;

  /*@member If SDF is shared by particles of a body.*/
  uint  collisionDataIsShared;
  /*@member Shared radius.*/
  ParticleCollisionData sharedCollisionData;
};

typedef struct ParticleSharedData_t ParticleSharedData;


/*
@struct Base data for a particle.
*/
struct DEFAULT_ALIGN ParticleStruct_t
{
  union
  {
    struct
    {
      float3  position;
    };
    struct
    {
      uint    reserved[3];
      float   radius;
    };
  };
};

typedef struct ParticleStruct_t ParticleStruct;


struct DEFAULT_ALIGN ParticleRigidData_t
{
  union
  {
    struct
    {
      float3  initialComOffset;
    };
    struct
    {
      uint    reserved[4];
    };
  };
};

typedef struct ParticleRigidData_t ParticleRigidData;


struct DEFAULT_ALIGN ParticleAuxData_t
{
  float   invMass;
  float   radius;
};

typedef struct ParticleAuxData_t ParticleAuxData;


struct DEFAULT_ALIGN ParticleDifferential_t
{
  float3  velocity;
};

typedef struct ParticleDifferential_t ParticleDifferential;


#ifdef COMPUTE_SHADER_SCOPE

float getInvMassUsingDeviceAux(const Thread ParticleSharedData* particleSharedData, const Device ParticleAuxData* particleAuxData, const uint index)
{
  if (particleSharedData->invMassIsShared)
  {
    return particleSharedData->sharedInvMass;
  }
  return particleAuxData[index].invMass;
}

float getInvMassUsingThreadAux(const Thread ParticleSharedData* particleSharedData, const Thread ParticleAuxData* particleAuxData)
{
  if (particleSharedData->invMassIsShared)
  {
    return particleSharedData->sharedInvMass;
  }
  return particleAuxData->invMass;
}

ParticleCollisionData getSDFUsingDeviceCollision(const Thread ParticleSharedData* particleSharedData, const Device ParticleCollisionData* particleCollisionData, const uint index)
{
  if (particleSharedData->collisionDataIsShared)
  {
    return particleSharedData->sharedCollisionData;
  }
  return particleCollisionData[index];
}

/*
@struct Uncompressed identity data for directl use at runtime.
*/
struct ParticleNodeIdentity_t
{
  uint entityId;
  uint instanceId;
  uint solverType;
};

typedef struct ParticleNodeIdentity_t ParticleNodeIdentity;

inline ParticleNodeIdentity uncompressToNodeIdentity(const IdentityInfo identity)
{
  ParticleNodeIdentity nodeIdentity;

  nodeIdentity.entityId = getEntityId(identity);
  nodeIdentity.instanceId = getInstanceId(identity);
  nodeIdentity.solverType = getSolverType(identity);

  return nodeIdentity;
}

/*
@struct Node index for various use.
*/
struct ParticleNodeLocator_t
{
  /*@member Offset to first node of the entity instance, in the physics system.*/
  uint absoluteNodeOffset;
  /*@member Index to this node's property in the entity instance ,in the physics system.*/
  uint absoluteNodeIndex;
  /*@member Index to this node's shader property in the entity, in the physics system.*/
  uint commonNodeIndex;
};

typedef struct ParticleNodeLocator_t ParticleNodeLocator;

inline ParticleNodeLocator getNodeLocator(const uint nodeIndex, const uint partitionInstanceOffset, const PartitionInfo nodeEntityLocation)
{
  ParticleNodeLocator locator;

  uint relativeNodeIndex;
  locator.absoluteNodeOffset = partitionInstanceOffset;
  relativeNodeIndex = nodeIndex % nodeEntityLocation.count;
  locator.absoluteNodeIndex = partitionInstanceOffset + relativeNodeIndex;
  locator.commonNodeIndex = nodeEntityLocation.offset + relativeNodeIndex;

  return locator;
}

#endif

struct DEFAULT_ALIGN PhySystemOffsets_t
{
  uint globalNodeOffset;
  uint globalInstanceOffset;
  uint globalSolverOffset;
};

typedef struct PhySystemOffsets_t PhySystemOffsets;

#endif