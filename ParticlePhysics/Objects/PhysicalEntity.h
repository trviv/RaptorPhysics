#ifndef __PHYSICAL_ENTITY
#define __PHYSICAL_ENTITY

#include "PhysicsSystem.h"

/// Class representing base entity
class PhysicalEntity
{
protected:
  //DistanceConstrain constrain;
  std::vector<unsigned __int32> connection_elements;
  Vertex  disp_vertex;
  Shader  disp_shader;
  Face    disp_elements;
  CudaGLPlug  plug;

public:
  virtual void init(const Matrix4& transform, const float dim[],
    const Counter subdivision[]) = 0;
  virtual void render() = 0;
  virtual void step() = 0;
};

#endif