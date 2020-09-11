#ifndef READER
#define READER

#include "UnifiedPhysics.h"
#include "RayTracing.h"
using namespace tinyxml2;

/*!
@class Base class to read contents of a scene descriptor file.
*/
class Reader
{
protected:

  static void readIntArray(int* dest, const char* source);

  static void readFloatArray(float* dest, const char* source);

public:

  virtual bool readFile(ComputeInterface* compute, PhysicsSystem* physicsSystem, Window* renderer, RayTracingSystem* rayTracingSystem, const char fname[]) = 0;
};

#endif

