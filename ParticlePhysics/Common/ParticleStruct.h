#ifndef PARTICLE_STRUCT_H
#define PARTICLE_STRUCT_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Core.h>
#endif

#define PHYSICS_SOLVER_ID_MASK    0xF0000000
#define PHYSICS_SOLVER_ID_SHIFT   28
#define PHYSICS_ENTITY_ID_MASK    0x0FFF0000
#define PHYSICS_ENTITY_ID_SHIFT   16
#define PHYSICS_INSTANCE_ID_MASK  0x0000FFFF

#pragma pack(push, 4)

/*
@struct Structor to uniquely represent a physical entity.
*/
struct ALIGN(4) IdentityInfo_t
{
  uint identity;
};

typedef struct IdentityInfo_t IdentityInfo;
typedef struct IdentityInfo_t PhysicsEntityId;

#ifndef COMPUTE_SHADER_SCOPE
static void resetIdentity(IdentityInfo& identity)
{
  identity.identity = -1;
}

static void setEntityId(IdentityInfo& identity, uint solver, uint entityId)
{
  identity.identity = (identity.identity & PHYSICS_INSTANCE_ID_MASK) |
    ((solver << PHYSICS_SOLVER_ID_SHIFT) & PHYSICS_SOLVER_ID_MASK) |
    ((entityId << PHYSICS_ENTITY_ID_SHIFT) & PHYSICS_ENTITY_ID_MASK);
}

static void setInstanceId(IdentityInfo& identity, uint instanceId)
{
  identity.identity = (identity.identity & (-1 ^ PHYSICS_INSTANCE_ID_MASK)) | (instanceId & PHYSICS_INSTANCE_ID_MASK);
}
#endif

static uint getInstanceId(const IdentityInfo particleIdentity)
{
  return particleIdentity.identity & PHYSICS_INSTANCE_ID_MASK;
}

static uint getEntityId(const IdentityInfo particleIdentity)
{
  return (particleIdentity.identity & PHYSICS_ENTITY_ID_MASK) >> PHYSICS_ENTITY_ID_SHIFT;
}

static uint getSolverType(const IdentityInfo particleIdentity)
{
  return (particleIdentity.identity & PHYSICS_SOLVER_ID_MASK) >> PHYSICS_SOLVER_ID_SHIFT;
}


/*
@struct Allocation data shared by all the particles of an entity.
*/
struct ALIGN(8) GroupData_t
{
  uint    minIdentity;
  uint    maxIdentity;
};

typedef struct GroupData_t GroupData;


/*
@struct Allocation data shared by all the particles of an entity.
*/
struct DEFAULT_ALIGN EntityLocation_t
{
  PartitionInfo node;
  PartitionInfo connection;
};

typedef struct EntityLocation_t EntityLocation;


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
      uint          reserved[3];
      IdentityInfo  identity;
    };
    struct // for rendering
    {
      uint    reserved2[3];
      float   radius;
    };
  };
};

typedef struct ParticleStruct_t ParticleStruct;


/*
@struct Collision data for each particle.
*/
struct DEFAULT_ALIGN ParticleCollisionData_t
{
  union
  {
    struct
    {
      float3  initialSdfGradient;
    };
    struct
    {
      float   reserved1[3];
      float   radius;
    };
  };
  union
  {
    struct
    {
      float3  transformedSdfGradient;
    };
    struct
    {
      float   reserved2[3];
      float   invMass;
    };
  };
};

typedef struct ParticleCollisionData_t ParticleCollisionData;


#define PARTICLE_SHARED_DATA_MASS_MASK      0x1
#define PARTICLE_SHARED_DATA_RADIUS_MASK    0x2
#define PARTICLE_SHARED_DATA_COLLISION_MASK 0x4

/*
@struct Data shared by all the particles of an entity.
*/
struct DEFAULT_ALIGN ParticleSharedData_t
{
  /*@member Mask for shared properties.*/
  uint  isSharedMask;
  /*@member Shared inverse mass.*/
  float sharedInvMass;
  /*@member Shared radius.*/
  float sharedRadius;
  /*@member Stiffness for spring constraint.*/
  float stiffness;
  /*@member Viscosity for fluid constraint.*/
  float viscosity;
  /*@member Velocity damping.*/
  float velocityDamping;
  /*@member Velocity damping.*/
  float collisionDamping;
  /*@member Kinetic friction coefficient.*/
  float kineticFrictionCoef;
  /*@member Static friction coefficient.*/
  float staticFrictionCoef;

  uint  padding[3];

  /*@member Shared collision data.*/
  ParticleCollisionData sharedCollisionData;
};

typedef struct ParticleSharedData_t ParticleSharedData;

#ifndef COMPUTE_SHADER_SCOPE
/*@function If mass is shared by particles of a body.*/
static void setInvMassIsShared(ParticleSharedData& data, bool isShared)
{
  data.isSharedMask = (data.isSharedMask & (-1 ^ PARTICLE_SHARED_DATA_MASS_MASK)) | (isShared ? PARTICLE_SHARED_DATA_MASS_MASK : 0);
}

/*@function If radius is shared by particles of a body.*/
static void setRadiusIsShared(ParticleSharedData& data, bool isShared)
{
  data.isSharedMask = (data.isSharedMask & (-1 ^ PARTICLE_SHARED_DATA_RADIUS_MASK)) | (isShared ? PARTICLE_SHARED_DATA_RADIUS_MASK : 0);
}

/*@function If SDF is shared by particles of a body.*/
static void setCollisionDataIsShared(ParticleSharedData& data, bool isShared)
{
  data.isSharedMask = (data.isSharedMask & (-1 ^ PARTICLE_SHARED_DATA_COLLISION_MASK)) | (isShared ? PARTICLE_SHARED_DATA_COLLISION_MASK : 0);
}
#endif


#ifdef COMPUTE_SHADER_SCOPE

bool getInvMassIsShared(const Thread ParticleSharedData* sharedData)
{
  return (sharedData->isSharedMask & PARTICLE_SHARED_DATA_MASS_MASK) > 0;
}

bool getRadiusIsShared(const Thread ParticleSharedData* sharedData)
{
  return (sharedData->isSharedMask & PARTICLE_SHARED_DATA_RADIUS_MASK) > 0;
}

bool getCollisionDataIsShared(const Thread ParticleSharedData* sharedData)
{
  return (sharedData->isSharedMask & PARTICLE_SHARED_DATA_COLLISION_MASK) > 0;
}

#endif

/*
@struct Data for rigid solver particle.
*/
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


/*
@struct Additional data for particle.
*/
struct ALIGN(4) ParticleAuxData_t
{
  float   invMass;
  float   radius;
};

typedef struct ParticleAuxData_t ParticleAuxData;


/*
@struct Particle differential data.
*/
struct DEFAULT_ALIGN ParticleDifferential_t
{
  float3  velocity;
};

typedef struct ParticleDifferential_t ParticleDifferential;

#ifdef COMPUTE_SHADER_SCOPE

float getInvMassUsingDeviceAux(const Thread ParticleSharedData* particleSharedData, const Device ParticleAuxData* particleAuxData, const uint index)
{
  if (getInvMassIsShared(particleSharedData))
  {
    return particleSharedData->sharedInvMass;
  }
  return particleAuxData[index].invMass;
}

float getInvMassUsingThreadAux(const Thread ParticleSharedData* particleSharedData, const Thread ParticleAuxData* particleAuxData)
{
  if (getInvMassIsShared(particleSharedData))
  {
    return particleSharedData->sharedInvMass;
  }
  return particleAuxData->invMass;
}

ParticleCollisionData getSDFUsingDeviceCollision(const Thread ParticleSharedData* particleSharedData, const Device ParticleCollisionData* particleCollisionData, const uint index)
{
  if (getCollisionDataIsShared(particleSharedData))
  {
    return particleSharedData->sharedCollisionData;
  }
  return particleCollisionData[index];
}

float getRadiusUsingDeviceAux(const Thread ParticleSharedData* particleSharedData, const Device ParticleAuxData* particleAuxData, const uint index)
{
  if (getRadiusIsShared(particleSharedData))
  {
    return particleSharedData->sharedRadius;
  }
  return particleAuxData[index].radius;
}

/*
@struct Uncompressed identity data for directl use at runtime.
*/
struct ALIGN(4) ParticleNodeIdentity_t
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
struct ALIGN(4) ParticleNodeLocator_t
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
  relativeNodeIndex = (nodeIndex - partitionInstanceOffset) % nodeEntityLocation.count;
  locator.absoluteNodeIndex = partitionInstanceOffset + relativeNodeIndex;
  locator.commonNodeIndex = nodeEntityLocation.offset + relativeNodeIndex;

  return locator;
}

#endif

struct ALIGN(4) PhySystemOffsets_t
{
  uint globalNodeOffset;
  uint globalInstanceOffset;
  uint globalSolverOffset;
};

typedef struct PhySystemOffsets_t PhySystemOffsets;

#pragma pack(pop)

#endif
