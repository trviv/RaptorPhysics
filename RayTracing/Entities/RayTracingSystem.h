#ifndef SCENE_H
#define SCENE_H

#include <Core.h>
#include <Light/Light.h>
#include <Camera/Camera.h>
#include <Acceleration/AccelerationDataStruct.h>
#include <Acceleration/BoundingVolumeHierarchyADS.h>
#include <Material/Material.h>

#define RAY_TRACING_SYSTEM_ARRAY_COUNT 2

/*!
@class Class representing a ray tracing system.
*/
class RayTracingSystem : protected ShaderEntity
{
  friend class ReaderScene;

protected:
  static uint rayComputeUtilId[RayStructTypeMax];
  static uint maxPrimIndex;
  static bool copyPrimitiveData;

  uint maxIterations;

  ComputeInterface*     compute;
  RayTracingAllocator*  allocator;

  ComputeKernel collectPrimitives;
  ComputeKernel transformPrimitives;
  ComputeKernel accumulateColor;
  ComputeKernel shadeIntersectionKernels[RayStructTypeMax][HitStructTypeMax];
  ComputeKernel processShadowRaysKernels[RayStructTypeMax][HitStructTypeMax];
  ComputeKernel reorderRaysKernels[RayStructTypeMax];
  ComputeKernel updateCameraKernel;

  vector<vector<RayTracingEntity*>> entitiyInstances;
  vector<RayTracingEntity*>         registeredEntities;
  vector<RayTracingEntity*>         registeredPrimitives[RTPrimitiveCount];

  /*!@member Final color output.*/
  DeviceArray<colorType4> colorOutputBuffer;
  DeviceArray<colorType4> accumulatedColorBuffer;

  /*!@member Camera used in the scene.*/
  Camera* camera;

  /*!@member Latest Camera used in the scene.*/
  DeviceArray<CameraStruct>* currentCamera;

  /*!@member Lights in the scene.*/
  DeviceArray<LightStruct> lights;

  /*!@member Materials in the scene.*/
  DeviceArray<MaterialStruct> materials;

  /*!@member Ray buffer for the scene.*/
  DeviceArray<uint> rays[RAY_TRACING_SYSTEM_ARRAY_COUNT+1];

  /*!@member Ray buffer for the scene.*/
  DeviceArray<uint> shadowRays[2];

  /*!@member Ray hit information buffer for the scene.*/
  DeviceArray<uint> hits;

  /*!@member Buffer holding indirect counts.*/
  DeviceArray<uint> indirectCount;

  /*!@member Composite array containing all positions.*/
  DeviceArray<PrimitiveStruct> vertexArray;

  /*!@member Composite array containing all primitive attributes.*/
  DeviceArray<PrimitiveAttrib> attributeArray;

  /*!@member Composite array containing all vertex attributes.*/
  DeviceArray<VertexAttrib>    vertexAttributeArray;

  DeviceArray<RTSystemSettings> systemSettings;

  DeviceArray<uint> randomUints;

  /*!@member Acceleration struct for the system.*/
  AccelerationDataStruct* accelerationStruct;

  uint newEntityId();

  uint newEntityInstanceId(uint entityIndex);

  /*!@function Register primitive from entity attributes.*/
  void registerPrimitive(RayTracingEntityType type, RayTracingEntity* entity);

  void composePrimitiveArray();

public:

  RayTracingSystem();

  ~RayTracingSystem();

  bool isAvailable();

  /*!@function Initialize the system based on max number of rays.*/
  void init(ComputeInterface* compute, const uint maxRays);

  /*!@function Process all registered entities and properties, commit to the device memory.*/
  void commit();

  /*!@function Get primitives in ray tracing system.*/
  uint getPrimCount()const;

  /*!@function Get RT system's camera.*/
  const Camera& getCameraStruct()const;

  /*!@function Get color buffer.*/
  const DeviceArray<colorType4>& getColorOutputBuffer()const;

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

  void render(bool updatePrimitives);
};

#endif
