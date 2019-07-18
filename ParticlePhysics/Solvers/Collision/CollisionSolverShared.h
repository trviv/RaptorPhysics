#ifndef COLLISION_SOLVER_SHARED_H
#define COLLISION_SOLVER_SHARED_H

#ifndef COMPUTE_SHADER_SCOPE

#include "../../Common/ParticleStruct.h"
#include "../SharedAllocator.h"

#endif

#pragma pack(push, 4)

/*
@struct Bounding volume hierarchy leaf data.
*/
struct ALIGN(8) BVHLeafInfo_t
{
  uint mortonCode;
  uint index;
};

typedef struct BVHLeafInfo_t BVHLeafInfo;

/*
@struct Bounding volume hierarchy internal node data.
*/
struct ALIGN(4) BVHNodeInfo_t
{
  uint child[2];
  uint parent;
};

typedef struct BVHNodeInfo_t BVHNodeInfo;

/*
@struct Axis aligned bounding box data.
*/
struct DEFAULT_ALIGN XAB_t
{
  union
  {
    float3  min;
    float   reserved1[4];
  };
  union
  {
    float3  max;
    float   reserved2[4];
  };
};

typedef struct XAB_t XAB;

#define mergeXAB(a, b)  { (a)->min = min((a)->min, (b)->min); (a)->max = max((a)->max, (b)->max);}
#define divXAB(a, b)    { (a)->min /= (*b); (a)->max /= (*b);}
#define clearXAB(a, b)  { (a)->min = INFINITY; (a)->max = -INFINITY;}

#pragma pack(pop)

#ifdef COMPUTE_SHADER_SCOPE

uint arrange32Bits(uint x)
{
  //........ ........ ......12 3456789A  //x
  //....1..2 ..3..4.. 5..6..7. .8..9..A  //x after interleaving bits

  //......12 3456789A ......12 3456789A  //x ^ (x << 16)
  //11111111 ........ ........ 11111111  //0x FF 00 00 FF
  //......12 ........ ........ 3456789A  //x = (x ^ (x << 16)) & 0xFF0000FF;

  //......12 ........ 3456789A 3456789A  //x ^ (x <<  8)
  //......11 ........ 1111.... ....1111  //0x 03 00 F0 0F
  //......12 ........ 3456.... ....789A  //x = (x ^ (x <<  8)) & 0x0300F00F;

  //..12..12 ....3456 3456.... 789A789A  //x ^ (x <<  4)
  //......11 ....11.. ..11.... 11....11  //0x 03 0C 30 C3
  //......12 ....34.. ..56.... 78....9A  //x = (x ^ (x <<  4)) & 0x030C30C3;

  //....1212 ..3434.. 5656..78 78..9A9A  //x ^ (x <<  2)
  //....1..1 ..1..1.. 1..1..1. .1..1..1  //0x 09 24 92 49
  //....1..2 ..3..4.. 5..6..7. .8..9..A  //x = (x ^ (x <<  2)) & 0x09249249;

  //........ ........ ......11 11111111  //0x000003FF

  x = (x ^ (x << 16)) & 0xFF0000FF;
  x = (x ^ (x << 8)) & 0x0300F00F;
  x = (x ^ (x << 4)) & 0x030C30C3;
  x = (x ^ (x << 2)) & 0x09249249;

  return x;
}

#define COLLISION_COMPONENT_MORTON_CODE_MASK 1023

uint get32BitMortonCode(const int3 quantizedPosition)
{
  const uint x = quantizedPosition.x & COLLISION_COMPONENT_MORTON_CODE_MASK;
  const uint y = quantizedPosition.y & COLLISION_COMPONENT_MORTON_CODE_MASK;
  const uint z = quantizedPosition.z & COLLISION_COMPONENT_MORTON_CODE_MASK;

  return arrange32Bits(x) | (arrange32Bits(y) << 1) | (arrange32Bits(z) << 2);
}

//#define MARK_COLLIDED_PARTICLES

/*
 @kernel Apply boundary constrain.
 @param particles Initial particle buffer.
 @param particles2 Secondary particle buffer.
 @param nodeLocator Node locator for base particle being processed.
 @param collisionData Particle SDF mass and radius data.
*/
void boundaryCollision(
  Thread ParticleStruct*      particle,
  Device ParticleStruct*      particles2,
  const ParticleNodeLocator   nodeLocator,
  const ParticleCollisionData collisionData)
{
  if (collisionData.invMass)  // only if movable
  {
    float dely = 0.f;

    if (particle->position.y <= -0.f)
    {
      dely = 0.f - particle->position.y;

      particle->position.y += dely;
      if (particles2)
      {
        particles2[nodeLocator.absoluteNodeIndex].position.y += dely;
      }
    }
  }
}

// function to process particle collision
inline float3 processParticleCollision(
  const ParticleStruct currentParticle,
  const ParticleStruct otherParticle,
  const ParticleCollisionData collisionData,
  const uint currentNodeIndex,
  const float sdfMagnitude,
#ifdef MARK_COLLIDED_PARTICLES
  Device ParticleCollisionData* particleCollisionData,
  Thread bool* collided,
#else
  const Device ParticleCollisionData* particleCollisionData,
#endif
  Thread ushort* collisionCount)
{
  if (otherParticle.identity.identity != currentParticle.identity.identity)
  {
    const ParticleCollisionData collisionData2 = particleCollisionData[currentNodeIndex];

    // skip if the base and the batch particle are of the same object
    const float3 distanceVector = otherParticle.position - currentParticle.position;
    //        const float3 distanceVector = currentParticle->position - otherParticle.position;
    const float actualDistance = dot(distanceVector, distanceVector);

#ifdef MARK_COLLIDED_PARTICLES
    const float allowedDistance = sqr(fabs(collisionData2.radius) + fabs(collisionData.radius));
#else
    const float allowedDistance = sqr(collisionData2.radius + collisionData.radius);
#endif

    // if overlapping
    if (actualDistance < allowedDistance)
    {
      const float sdfMagnitude2 = length(collisionData2.transformedSdfGradient);
      float3 normal = select(-collisionData2.transformedSdfGradient, collisionData.transformedSdfGradient, constructUint3(sdfMagnitude < sdfMagnitude2));
      const float collDot = dot(normal, distanceVector);

      //          if (collDot < 0.f)
      //          {
      //            normal = distanceVector - (2.f * collDot) * normal;
      //          }

#ifdef MARK_COLLIDED_PARTICLES
      *collided = true;
#endif
      (*collisionCount)++;
      return normal * (collisionData.invMass / (collisionData.invMass + collisionData2.invMass));
    }
  }

  return constructFloat3(0.f);
}

#endif

#endif
