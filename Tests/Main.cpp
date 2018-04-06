#include "Core.h"
#include "../ParticlePhysics/Common/ParticleStruct.h"

static ComputeInterface* compute;

/*void testEquation(ComputeInterface* compute)
{
SharedAllocator allocator(compute);
allocator.constrainAllocator.create(1024, 1024 * 16);
allocator.particleAllocator.create(1024);

PartitionInfo entityLocation;
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

cons.commit(entityLocation);

cons.addConnection(0, 0, 2);
cons.addConnection(1, 1, 7);
cons.addConnection(0, 1, 1);
cons.addConnection(1, 0, 5);

cons.setConstant(0, 11);
cons.setConstant(1, 13);

cons.commit(entityLocation);

cons.solve();
}*/

void test1DMean(ComputeInterface* compute)
{
  printf("\nTesting 1D mean:\n");

  DeviceArray<float> data(compute, NULL, true);
  const int elements = 1234567;
  float sum = 0;

  data.host()->reserve(elements);
  for (int i = 0; i < elements; i++)
  {
    data.host()->push_back(float(rand()) / RAND_MAX);
    sum += data.host()->at(i);
  }
  sum /= elements;

  data.syncDevice();

  map<ComputeUtilKey, string> utilSetting;
  utilSetting[ComputeUtilStructType] = "float";
  uint templateId = ComputeUtil::create(compute, utilSetting, NULL);

  ComputeUtil::get(templateId)->sum1D(compute, data.device(), elements, true);

  data.syncHost(0, 1);
  compute->sync();

  std::cout << sum << " " << data.host()->at(0) << "\n";
  assert(abs(sum - data.host()->at(0)) <= .00001f);

  printf("1D array mean test passed!\n");
}

void testRegular2DMean(ComputeInterface* compute)
{
  printf("\nTesting regular 2D mean:\n");

  DeviceArray<ParticleStruct> particles(compute, NULL, true);
  DeviceArray<PartitionInfo>  partitions(compute, NULL, true);

  uint width = 83;
  const uint parts = 625;
  const int elements = width * parts;
  Real3 sum = 0;

  for (int i = 0; i < parts; i++)
  {
    PartitionInfo section;
    section.offset = i * width;
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
      means.push_back(sum / float(width));
      sum = 0;
      sectionIndex++;
    }
    particlesHost.push_back(particle);
    sum += particle.position;
  }

  means.push_back(sum / float(width));

  particles.syncDevice();
  partitions.syncDevice();

  vector<string> includes = { "ParticleStruct.h" };
  map<ComputeUtilKey, string> utilSetting;
  utilSetting[ComputeUtilStructType] = "ParticleStruct";
  utilSetting[ComputeUtilStructMember] = "position";

  uint templateId = ComputeUtil::create(compute, utilSetting, &includes);

  ComputeUtil::get(templateId)->sumRegular2D(compute, particles.device(), elements, width, true);

  particles.syncHost();
  compute->sync();

  for (int i = 0; i < parts; i++)
  {
    if (abs(means[i][0] - particlesHost[i * width].position[0])>.00001f
      || abs(means[i][1] - particlesHost[i * width].position[1]) > .00001f
      || abs(means[i][2] - particlesHost[i * width].position[2]) > .00001f)
    {
      std::cout << i << " " << means[i] << " " << particlesHost[i * width].position << "\n";
      assert(0);
    }
  }
  printf("Regular 2D mean test passed!\n");
}

void testIrregular2DMean(ComputeInterface* compute)
{
  printf("\nTesting irregular 2D mean:\n");

  DeviceArray<ParticleStruct> particles(compute, NULL, true);
  DeviceArray<IdentityInfo>   particleIdentities(compute, NULL, true);
  DeviceArray<PartitionInfo>  partitions(compute, NULL, true);
  DeviceArray<uint>           partitionCount(compute, NULL, true);

  uint width = 0;
  const uint parts = 30;
  uint elements = 0;
  Real3 sum = 0;

  for (int i = 0; i < parts; i++)
  {
    PartitionInfo section;
    section.offset = i ? partitions.host()->at(i - 1).offset + width : 0;
    section.count = width + 1;
    partitions.host()->push_back(section);
    elements += width + 1;
    width++;
  }

  particles.host()->reserve(elements);
  partitionCount.host()->reserve(1);
  partitionCount.host()->push_back(parts);
  partitionCount.syncDevice();

  uint sectionIndex = 0;
  vector<PartitionInfo> &partitionsHost = *partitions.host();
  vector<ParticleStruct> &particlesHost = *particles.host();
  vector<Real3> means;

  for (uint i = 0; i < elements; i++)
  {
    ParticleStruct particle;
    particle.position = 1;
    if (sectionIndex < (partitionsHost.size() - 1) && i == partitionsHost[sectionIndex + 1].offset)
    {
      means.push_back(sum / float(partitionsHost[sectionIndex + 1].offset - partitionsHost[sectionIndex].offset));
      sum = 0;
      sectionIndex++;
    }

    IdentityInfo particleIdentity;
    particleIdentity.setInstanceId(sectionIndex);
    particleIdentities.host()->push_back(particleIdentity);

    particlesHost.push_back(particle);
    sum += particle.position;
  }

  means.push_back(sum / float(elements - partitionsHost[sectionIndex].offset));

  particles.syncDevice();
  partitions.syncDevice();
  particleIdentities.syncDevice();

  vector<string> includes = { "ParticleStruct.h" };
  map<ComputeUtilKey, string> utilSetting;

  utilSetting[ComputeUtilStructType] = "ParticleStruct";
  utilSetting[ComputeUtilStructMember] = "position";

  utilSetting[ComputeUtilIdentityStructType] = "IdentityInfo";
  utilSetting[ComputeUtilIdentityFunction] = "getInstanceId";

  uint templateId = ComputeUtil::create(compute, utilSetting, &includes);

  ComputeUtil::get(templateId)->sumIrregular2D(compute, particles.device(), particleIdentities.device(), partitions.device(), partitionCount.device(), elements, width, true);

  particles.syncHost();
  compute->sync();

  for (uint i = 0; i < partitionsHost.size(); i++)
  {
    if (abs(means[i][0] - particlesHost[partitionsHost[i].offset].position[0]) > .00001f
      || abs(means[i][1] - particlesHost[partitionsHost[i].offset].position[1]) > .00001f
      || abs(means[i][2] - particlesHost[partitionsHost[i].offset].position[2]) > .00001f)
    {
      std::cout << i << " " << means[i] << " " << particlesHost[partitionsHost[i].offset].position << "\n";
      assert(0);
    }
  }
  printf("Irregular 2D mean test passed!\n");

  printf("\nTesting consolidation:\n");

  DeviceArray<ParticleStruct> particlesConsolidated(compute, NULL, true);
  DeviceArray<uint> partitionsCount(compute, NULL, true);

  particlesConsolidated.resize(parts, false);
  partitionsCount.host()->push_back(parts);
  partitionsCount.syncDevice();

  ComputeUtil::get(templateId)->consolidateFromPartitions(compute,
    particles.device(), particlesConsolidated.device(), partitions.device(), partitionsCount.device(), parts);

  particlesConsolidated.syncHost();
  compute->sync();

  for (uint i = 0; i < particlesConsolidated.host()->size(); i++)
  {
    if (abs(means[i][0] - particlesConsolidated.host()->at(i).position[0]) > .00001f
      || abs(means[i][1] - particlesConsolidated.host()->at(i).position[1]) > .00001f
      || abs(means[i][2] - particlesConsolidated.host()->at(i).position[2]) > .00001f)
    {
      std::cout << i << " " << means[i] << " " << particlesConsolidated.host()->at(i).position << "\n";
      assert(0);
    }
  }
  printf("Consolidation test passed!\n");
}

/*void testPrefixSum1D(ComputeInterface* compute)
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

map<ComputeUtilKey, string> utilSetting;
utilSetting[ComputeUtilStructType] = "float";
uint templateId = ComputeUtil::create(compute, utilSetting, NULL);

ComputeUtil::get(templateId)->prefixSum1D(compute, data.device(), elements, true);

data.syncHost();
compute->sync();

for (uint i = 0; i < data.host()->size(); i++)
{
std::cout << original[i] << " " << prefixSum[i] << " " << data.host()->at(i) << "\n";
//assert(prefixSum[i] == data.host()->at(0));
}
}*/

/*void testSectionOffsets(ComputeInterface* compute)
{
printf("\nTesting section offset:\n");

vector<uint>                offsetOutput;
DeviceArray<uint>           offsets(compute, NULL, true);
DeviceArray<uint>           offsetCount(compute, NULL, true);
DeviceArray<PartitionInfo>  partitions(compute, NULL, true);

const uint parts = 86878;
uint nodes = 2;
uint instanceCount = 1;
uint totalNodes = 0;
uint totalInstances = 0;

for (int i = 0; i < parts; i++)
{
PartitionInfo section;
section.offset = i ? (partitions.host()->at(i - 1).offset + partitions.host()->at(i - 1).count) : 0;
section.count = nodes * instanceCount;
//section.identity.setIdentity(instanceCount, 0);

partitions.host()->push_back(section);
for (uint j = 0; j < instanceCount; j++)
{
offsetOutput.push_back(totalNodes);
totalNodes += nodes;
}
totalInstances += instanceCount;
nodes++;
if (i % 505 == 0)
{
instanceCount += 1;
}
}

offsets.resize(totalInstances, false);
offsetCount.resize(1, false);
partitions.syncDevice();

vector<string>              includes;
map<ComputeUtilKey, string> utilSetting;
includes.push_back("ParticleStruct.h");
uint templateId = ComputeUtil::create(compute, utilSetting, &includes);

ComputeUtil::get(templateId)->createSectionOffsets(compute, offsets.device(), offsetCount.device(), partitions.device(), parts);

offsets.syncHost();
offsetCount.syncHost();
compute->sync();

printf("Section offset count %d\n", offsetCount.host()->at(0));
for (uint i = 0; i < offsetOutput.size(); i++)
{
if (offsetOutput[i] != offsets.host()->at(i))
{
std::cout << offsetOutput[i] << " " << offsets.host()->at(i) << "\n";
assert(offsetOutput[i] == offsets.host()->at(i));
}
}
printf("Section offset test passed!\n");
}*/

int main(int argc, char** argv)
{
  compute = new ComputeInterface();
  compute->create(1);

  //testSectionOffsets(compute);
  test1DMean(compute);
  testRegular2DMean(compute);
  testIrregular2DMean(compute);
  //testPrefixSum1D(compute);
  //testEquation(compute);

  return 0;
}