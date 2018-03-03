#ifndef PARTICLE_STRUCT_H
#define PARTICLE_STRUCT_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Core.h>
typedef Real3 float3;
typedef Real3 float4;
#endif

#define DEVICE_HEADER_NODE        0
#define DEVICE_HEADER_CONNECTION  1

/*
@struct Allocation data shared by all the particles of an entity.
*/
struct DEFAULT_ALIGN SectionData_t
{
  /*@member Offsets.*/
  uint    offsets[2];
  /*@member Counts.*/
  uint    counts[2];
};

typedef struct SectionData_t SectionData;


/*
@struct Allocation data shared by all the particles of an entity.
*/
struct DEFAULT_ALIGN GroupData_t
{
  uint    minIdentity;
  uint    maxIdentity;
};

typedef struct GroupData_t GroupData;


/*
@struct Data shared by all the particles of an entity.
*/
struct DEFAULT_ALIGN ParticleSharedData_t
{
  /*@member If mass is shared by particles of a body.*/
  uint    invMassIsShared;
  /*@member Shared inverse mass.*/
  float   sharedInvMass;

  /*@member If radius is shared by particles of a body.*/
  uint    radiusIsShared;
  /*@member Shared radius.*/
  float   sharedRadius;

  /*@member Stiffness for spring constraint.*/
  float   stiffness;
  /*@member Viscosity for fluid constraint.*/
  float   viscosity;
  /*@member Velocity damping.*/
  float   velocityDamping;
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
      uint    reserved[3], identity;
    };
  };
};

typedef struct ParticleStruct_t ParticleStruct;


struct DEFAULT_ALIGN ParticleRigidData_t
{
  float3  initialComOffset;
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


#define PARTICLE_ENTITY_ID_MASK     0xFFFFFFF
#define PARTICLE_INSTANCE_ID_SHIFT  28

static uint getEntityId(const uint particleIdentity)
{
  return particleIdentity&PARTICLE_ENTITY_ID_MASK;
}

static uint getInstanceId(const uint particleIdentity)
{
  return particleIdentity >> PARTICLE_INSTANCE_ID_SHIFT;
}


#ifdef COMPUTE_SHADER_SCOPE

float getInvMass(const Thread ParticleSharedData* particleSharedData, const Device ParticleAuxData* particleAuxData, const uint index)
{
  if (particleSharedData->invMassIsShared)
  {
    return particleSharedData->sharedInvMass;
  }
  return particleAuxData[index].invMass;
}

#endif

#endif