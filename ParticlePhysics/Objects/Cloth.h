#ifndef __CLOTH
#define __CLOTH

//#include "PhysicsSystem.h"
#include "PhysicalEntity.h"

/// Class representing a cloth
class Cloth : public PhysicalEntity
{
public:
  /*
  DistanceConstrain constrain;
  std::vector<int> line_elements;
  Vertex disp_vertex;
  Shader disp_shader;
  Face   disp_elements;
  CudaGLPlug plug;
  */
public:
  void init(const Matrix4& transform, const float dim[],
    const Counter subdivision[]);
  void render();
  void step();
};

#endif