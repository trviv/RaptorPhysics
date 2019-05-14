#ifndef PARTICLE_STRUCT_H
#define PARTICLE_STRUCT_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Core.h>
typedef Real3 float3;
typedef Real3 float4;
#endif

#define PHYSICS_SOLVER_ID_MASK    0xF0000000
#define PHYSICS_SOLVER_ID_SHIFT   28
#define PHYSICS_ENTITY_ID_MASK    0x0FFF0000
#define PHYSICS_ENTITY_ID_SHIFT   16
#define PHYSICS_INSTANCE_ID_MASK  0x0000FFFF


struct IdentityInfo_t
{
  uint identity;

#ifndef COMPUTE_SHADER_SCOPE

  IdentityInfo_t()
  {
    identity = -1;
  }

  void setEntityId(uint solver, uint entityId)
  {
    identity = (identity & PHYSICS_INSTANCE_ID_MASK) |
      ((solver << PHYSICS_SOLVER_ID_SHIFT) & PHYSICS_SOLVER_ID_MASK) |
      ((entityId << PHYSICS_ENTITY_ID_SHIFT) & PHYSICS_ENTITY_ID_MASK);
  }

  void setInstanceId(uint instanceId)
  {
    identity = (identity & (-1 ^ PHYSICS_INSTANCE_ID_MASK)) | (instanceId & PHYSICS_INSTANCE_ID_MASK);
  }

#endif
};

typedef struct IdentityInfo_t IdentityInfo;
typedef struct IdentityInfo_t PhysicsEntityId;

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


#pragma pack(push)
#pragma pack(4)

/*
@struct Base data for a particle.
*/
struct ParticleStruct_t
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

#ifndef COMPUTE_SHADER_SCOPE
  ParticleStruct_t()
  {}

  ParticleStruct_t(const ParticleStruct_t& ref)
  {
    *this = ref;
  }

  ParticleStruct_t& operator = (const ParticleStruct_t& ref)
  {
    position = ref.position;
    identity = ref.identity;
    return *this;
  }
#endif
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
      float reserved1[3];
      float radius;
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

#ifndef COMPUTE_SHADER_SCOPE
  ParticleCollisionData_t()
  {}

  ParticleCollisionData_t(const ParticleCollisionData_t& ref)
  {
    *this = ref;
  }

  ParticleCollisionData_t& operator = (const ParticleCollisionData_t& ref)
  {
    initialSdfGradient = ref.initialSdfGradient;
    radius = ref.radius;
    transformedSdfGradient = ref.transformedSdfGradient;
    invMass = ref.invMass;
    return *this;
  }
#endif
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

  uint  padding[2];

  /*@member Shared collision data.*/
  ParticleCollisionData sharedCollisionData;

#ifndef COMPUTE_SHADER_SCOPE

  ParticleSharedData_t()
  {
    isSharedMask = 0;
  }

  ParticleSharedData_t(const ParticleSharedData_t& ref)
  {
    *this = ref;
  }

  ParticleSharedData_t& operator = (const ParticleSharedData_t& ref)
  {
    isSharedMask = ref.isSharedMask;
    sharedInvMass = ref.sharedInvMass;
    sharedRadius = ref.sharedRadius;
    stiffness = ref.stiffness;
    viscosity = ref.viscosity;
    velocityDamping = ref.velocityDamping;
    sharedCollisionData = ref.sharedCollisionData;

    return *this;
  }

  /*@function If mass is shared by particles of a body.*/
  void setInvMassIsShared(bool isShared)
  {
    isSharedMask = (isSharedMask & (-1 ^ PARTICLE_SHARED_DATA_MASS_MASK)) | (isShared ? PARTICLE_SHARED_DATA_MASS_MASK : 0);
  }

  /*@function If radius is shared by particles of a body.*/
  void setRadiusIsShared(bool isShared)
  {
    isSharedMask = (isSharedMask & (-1 ^ PARTICLE_SHARED_DATA_RADIUS_MASK)) | (isShared ? PARTICLE_SHARED_DATA_RADIUS_MASK : 0);
  }

  /*@function If SDF is shared by particles of a body.*/
  void setCollisionDataIsShared(bool isShared)
  {
    isSharedMask = (isSharedMask & (-1 ^ PARTICLE_SHARED_DATA_COLLISION_MASK)) | (isShared ? PARTICLE_SHARED_DATA_COLLISION_MASK : 0);
  }

#endif
};

typedef struct ParticleSharedData_t ParticleSharedData;

#pragma pack(pop)


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

#ifndef COMPUTE_SHADER_SCOPE
  ParticleRigidData_t()
  {}

  ParticleRigidData_t(const ParticleRigidData_t& ref)
  {
    *this = ref;
  }

  ParticleRigidData_t& operator = (const ParticleRigidData_t& ref)
  {
    initialComOffset = ref.initialComOffset;
    return *this;
  }
#endif
};

typedef struct ParticleRigidData_t ParticleRigidData;


/*
@struct Additional data for particle.
*/
struct DEFAULT_ALIGN ParticleAuxData_t
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
  relativeNodeIndex = (nodeIndex - partitionInstanceOffset) % nodeEntityLocation.count;
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
