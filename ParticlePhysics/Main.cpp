#include "UnifiedPhysics.h"

#pragma comment(lib, "glew32.lib")

static ComputeInterface* compute;
static PhysicsSystem* physicsSystem;


void testEquation(ComputeInterface* compute)
{
  SharedAllocator allocator(compute);
  allocator.constrainAllocator.create(1024, 1024 * 16);
  allocator.particleAllocator.create(1024);

  LinearSolver<ushort, float, float> cons(compute, &allocator);

  cons.create(compute);

  cons.addConnection(0, 0, 10);
  cons.addConnection(1, 1, 11);
  cons.addConnection(2, 2, 10);
  cons.addConnection(3, 3, 8);

  cons.addConnection(0, 1, -1);
  cons.addConnection(0, 2, 2);

  cons.addConnection(1, 0, -1);
  cons.addConnection(1, 2, -1);
  cons.addConnection(1, 3, 3);

  cons.addConnection(2, 0, 2);
  cons.addConnection(2, 1, -1);
  cons.addConnection(2, 3, -1);

  cons.addConnection(3, 1, 3);
  cons.addConnection(3, 2, -1);

  cons.setConstant(0, 6);
  cons.setConstant(1, 25);
  cons.setConstant(2, -11);
  cons.setConstant(3, 15);

  cons.commit();

  cons.addConnection(0, 0, 2);
  cons.addConnection(1, 1, 7);
  cons.addConnection(0, 1, 1);
  cons.addConnection(1, 0, 5);

  cons.setConstant(0, 11);
  cons.setConstant(1, 13);

  cons.commit();

  cons.solve();
}
void test1DMean(ComputeInterface* compute)
{
  DeviceArray<float> data(compute, NULL, true);
  const int elements = 12345678;
  float sum = 0;

  data.host()->reserve(elements);
  for (int i = 0; i < elements; i++)
  {
    data.host()->push_back(float(rand()) / RAND_MAX);
    sum += data.host()->at(i);
  }
  sum /= elements;

  data.syncDevice();

  vector<ComputeUtilTuple> utilSetting;
  utilSetting.push_back(ComputeUtilTuple(ComputeUtilStructType, "float"));
  uint templateId = ComputeUtil::create(compute, utilSetting, NULL);

  ComputeUtil::get(templateId)->sum1D(compute, data.device(), elements, true);

  data.syncHost(0, 1);
  compute->sync();

  std::cout << sum << " " << data.host()->at(0) << "\n";
  assert(sum == data.host()->at(0));
}

void testRegular2DMean(ComputeInterface* compute)
{
  DeviceArray<ParticleStruct> particles(compute, NULL, true);
  DeviceArray<SectionData>    partitions(compute, NULL, true);

  uint width = 83;
  const uint parts = 625;
  const int elements = width * parts;
  Real3 sum = 0;

  for (int i = 0; i < parts; i++)
  {
    SectionData section;
    section.offsets[DEVICE_HEADER_NODE] = i * width;
    partitions.host()->push_back(section);
  }

  particles.host()->reserve(elements);

  int sectionIndex = 0;
  vector<ParticleStruct> &particlesHost = *particles.host();
  vector<Real3> means;

  for (int i = 0; i < elements; i++)
  {
    ParticleStruct particle;
    particle.position = 1;
    if (i && (i % width) == 0)
    {
      means.push_back(sum);
      sum = 0;
      sectionIndex++;
    }
    particle.identity = sectionIndex;
    particlesHost.push_back(particle);
    sum += particle.position;
  }

  means.push_back(sum);

  particles.syncDevice();
  partitions.syncDevice();

  vector<string> includes = { "ParticleStruct.h" };
  vector<ComputeUtilTuple> utilSetting;
  utilSetting.push_back(ComputeUtilTuple(ComputeUtilStructType, "ParticleStruct"));
  utilSetting.push_back(ComputeUtilTuple(ComputeUtilStructMember, "position"));

  uint templateId = ComputeUtil::create(compute, utilSetting, &includes);

  ComputeUtil::get(templateId)->sumRegular2D(compute, particles.device(), elements, width, true);

  particles.syncHost();
  compute->sync();

  for (int i = 0; i < parts; i++)
  {
    std::cout << means[i] << " " << particlesHost[i * width].position << "\n";
  }
}

void testIrregular2DMean(ComputeInterface* compute)
{
  DeviceArray<ParticleStruct> particles(compute, NULL, true);
  DeviceArray<SectionData>    partitions(compute, NULL, true);

  uint width = 0;
  const uint parts = 46;
  const int elements = (parts * (parts + 1)) >> 1;// width * parts;
  Real3 sum = 0;

  for (int i = 0; i < parts; i++)
  {
    SectionData section;
    section.offsets[DEVICE_HEADER_NODE] = width ? section.offsets[DEVICE_HEADER_NODE] + width : 0;
    partitions.host()->push_back(section);
    width++;
  }

  particles.host()->reserve(elements);

  int sectionIndex = 0;
  vector<SectionData> &partitionsHost = *partitions.host();
  vector<ParticleStruct> &particlesHost = *particles.host();
  vector<Real3> means;

  for (int i = 0; i < elements; i++)
  {
    ParticleStruct particle;
    particle.position = 1;
    if (sectionIndex < (partitionsHost.size() - 1) &&
      i == partitionsHost[sectionIndex + 1].offsets[DEVICE_HEADER_NODE])
    {
      means.push_back(sum);
      sum = 0;
      sectionIndex++;
    }
    particle.identity = sectionIndex;
    particlesHost.push_back(particle);
    sum += particle.position;
  }

  means.push_back(sum);

  particles.syncDevice();
  partitions.syncDevice();

  vector<string> includes = { "ParticleStruct.h" };
  vector<ComputeUtilTuple> utilSetting;
  utilSetting.push_back(ComputeUtilTuple(ComputeUtilStructType, "ParticleStruct"));
  utilSetting.push_back(ComputeUtilTuple(ComputeUtilStructMember, "position"));
  utilSetting.push_back(ComputeUtilTuple(ComputeUtilStructIdentity, "identity"));
  utilSetting.push_back(ComputeUtilTuple(ComputeUtilIndexStructType, "SectionData"));
  utilSetting.push_back(ComputeUtilTuple(ComputeUtilIndexStructMember, "offsets[DEVICE_HEADER_NODE]"));

  uint templateId = ComputeUtil::create(compute, utilSetting, &includes);

  ComputeUtil::get(templateId)->sumIrregular2D(compute, particles.device(), partitions.device(), elements, width, false);

  particles.syncHost();
  compute->sync();

  for (int i = 0; i < partitionsHost.size(); i++)
  {
    std::cout << means[i] << " " << particlesHost[partitionsHost[i].offsets[DEVICE_HEADER_NODE]].position << "\n";
  }
}

void testPrefixSum1D(ComputeInterface* compute)
{
  DeviceArray<float> data(compute, NULL, true);
  const int elements = 12;
  std::vector<float> prefixSum;
  std::vector<float> original;

  data.host()->reserve(elements);
  for (int i = 0; i < elements; i++)
  {
    data.host()->push_back(1.f);// float(rand()) / RAND_MAX);
    original.push_back(data.host()->at(i));
    prefixSum.push_back(i ? (prefixSum[i - 1] + data.host()->at(i)) : data.host()->at(i));
  }

  data.syncDevice();

  vector<ComputeUtilTuple> utilSetting;
  utilSetting.push_back(ComputeUtilTuple(ComputeUtilStructType, "float"));
  uint templateId = ComputeUtil::create(compute, utilSetting, NULL);

  ComputeUtil::get(templateId)->prefixSum1D(compute, data.device(), elements, true);

  data.syncHost();
  compute->sync();

  for (uint i = 0; i < data.host()->size(); i++)
  {
    std::cout << original[i] << " " << prefixSum[i] << " " << data.host()->at(i) << "\n";
    //assert(prefixSum[i] == data.host()->at(0));
  }
}

int main(int argc, char** argv)
{
  compute = new ComputeInterface();
  compute->create(1);

  //test1DMean(compute);
  //testRegular2DMean(compute);
  //testIrregular2DMean(compute);
  //testPrefixSum1D(compute);
  //testEquation(compute);

  //return 0;

  physicsSystem = new PhysicsSystem(compute);
  main_window = physicsSystem;
  main_window->init(argc, argv);

  real dim[3];

  Cloth* cloth = new Cloth();
  RigidBody* rigidBody = new RigidBody();

  dim[0] = 1;
  dim[1] = 1;
  dim[2] = 1;

  uint subdivision1[2] = { 8, 8 };
  cloth->init(Matrix4(), dim, subdivision1, 1);
  physicsSystem->registerEntity(cloth);

  uint subdivision2[3] = { 6, 6, 6 };
  rigidBody->init(Matrix4(), dim, subdivision2, 1);
  physicsSystem->registerEntity(rigidBody);

  main_window->start();

  delete physicsSystem;
  delete compute;

  return 0;
}