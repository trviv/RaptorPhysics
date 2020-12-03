#ifndef SCENE_H
#define SCENE_H

#include <Core.h>
#include <Light/Light.h>
#include <Camera/Camera.h>
#include <Acceleration/AccelerationDataStruct.h>
#include <Acceleration/BoundingVolumeHierarchyADS.h>
#include <Material/Material.h>

/*!
@class Class representing a ray tracing system.
*/
class RayTracingSystem : protected ShaderEntity
{
  friend class ReaderScene;

protected:
  static uint rayComputeUtilId[RayStructTypeMax];

  ComputeInterface*     compute;
  RayTracingAllocator*  allocator;

  ComputeKernel collectPrimitives;
  ComputeKernel shadeIntersectionKernels[RayStructTypeMax][HitStructTypeMax];
  ComputeKernel processShadowRaysKernels[RayStructTypeMax][HitStructTypeMax];

  vector<vector<RayTracingEntity*>> entities;

  vector<EntityPrimAttributes>      registeredPrimitives[RTPrimitiveCount];

  /*!@member Final color output.*/
  DeviceArray<uint> colorOutputBuffer;

  /*!@member Camera used in the scene.*/
  Camera* camera;

  /*!@member Lights in the scene.*/
  DeviceArray<LightStruct>  lights;

  /*!@member Materials in the scene.*/
  DeviceArray<MaterialStruct> materials;

  /*!@member Ray buffer for the scene.*/
  DeviceArray<uint>         rays;

  /*!@member Ray buffer for the scene.*/
  DeviceArray<uint>         shadowRays;

  /*!@member Ray hit information buffer for the scene.*/
  DeviceArray<uint>         hits;

  /*!@member Composite array containing all positions.*/
  DeviceArray<PrimitiveStruct>  vertexArray;

  /*!@member Composite array containing all attributes.*/
  DeviceArray<PrimitiveAttrib>  attributeArray;

  DeviceArray<RTSystemSettings> systemSettings;

  /*!@member Acceleration struct for the system.*/
  AccelerationDataStruct*   accelerationStruct;

  DeviceArray<uint>         validRayCount;

  uint newEntityId();

  uint newEntityInstanceId(uint entityIndex);

  /*!@function Register primitive from entity attributes.*/
  void registerPrimitive(RayTracingEntityType type, RayTracingEntity* entity);

  void composePrimitiveArray();

public:

  RayTracingSystem();

  ~RayTracingSystem();

  /*!@function Initialize the system based on max number of rays.*/
  void init(ComputeInterface* compute, const uint maxRays);

  /*!@function Process all registered entities and properties, commit to the device memory.*/
  void commit();

  /*!@function Get primitives in ray tracing system.*/
  uint getPrimCount()const;

  /*!@function Get RT system's camera.*/
  const Camera& getCameraStruct()const;

  const DeviceArray<uint>& getColorOutputBuffer()const;

  /*!@function Register a Material to the system.*/
  MaterialId registerMaterial(Material* material);

  /*!@function Register a Ray Tracing entity to the system.*/
  RayTracingEntityId registerEntity(RayTracingEntity* entity);

  /*!@function Instantiate an entity registered within the system.*/
  void addEntityInstance(const RayTracingEntityId& registeredEntityId, ushort instanceCount = 1, const Matrix4* instanceTransforms = NULL);

  /*!@function Register a Ray Tracing entity and instantiate.*/
  RayTracingEntityId registerAndInstantiateEntity(RayTracingEntity* entity, ushort instanceCount = 1, const Matrix4* instanceTransforms = NULL);

  /*!@function Update camera based on given matrices.*/
  void updateCamera(const real projectionMatrix[16], const real modelviewMatrix[16]);

  void render();
};

#endif
