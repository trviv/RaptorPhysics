#include "Core.h"
#include "../ParticlePhysics/Common/ParticleStruct.h"
#include <algorithm>

static ComputeInterface* compute;

void printStats(float mean, uint elements, uint rwCount, uint sizeOfElements)
{
  printf("Average time    : %f ms\n", mean);
  printf("Elts/sec        : %f M\n", elements * 1000.f / (mean * 1000 * 1000));
  printf("Bandwidth util  : %f GB/s\n", (rwCount * sizeOfElements * elements) * (1000.f / mean) / float(1024 * 1024 * 1024));
}

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

void testBandwidthRW(ComputeInterface* compute)
{
  printf("\nRead/Write bandwidth test:\n");

  DeviceArray<float> data(compute, NULL, false);
  const int elements = 1024 * 1024 * 32;

  data.resize(elements, false);

  map<ComputeUtilKey, string> utilSetting;
  utilSetting[ComputeUtilStructType] = "uint";
  uint templateId = ComputeUtil::create(compute, utilSetting, NULL);

  uint iterations = 10;

  ProfileManager::Reset();
  {
    ProfileBlock("R/W Bandwidth");
    for (uint i = 0; i < iterations; i++)
    {
      compute->copyBuffer(data.device(), data.device(), 0, 0, elements * sizeof(uint));
    }
  }
  compute->sync();

  float mean = ProfileManager::Get_Time_Since_Reset() / iterations;
  printStats(mean, elements, 2, sizeof(uint));
}

template<class DataType> void test1DMean(ComputeInterface* compute)
{
  printf("\nTesting 1D mean:\n");

  DeviceArray<DataType> data(compute, NULL, true);
  DeviceArray<DataType> backupData(compute, NULL, false);

  const int elements = 12345678;// 1024 * 1024 * 16;
  DataType sum = 0;
  uint iterations = 20;

  data.host()->reserve(elements);
  for (int i = 0; i < elements; i++)
  {
    data.host()->push_back(float(rand()) / RAND_MAX);
    sum += data.host()->at(i);
  }
  sum /= elements;

  backupData.resize(elements, false);
  data.syncDevice();
  compute->copyBuffer(data.device(), backupData.device(), 0, 0, elements * sizeof(DataType));

  map<ComputeUtilKey, string> utilSetting;
  utilSetting[ComputeUtilStructType] = "float";
  uint templateId = ComputeUtil::create(compute, utilSetting, NULL);
  ComputeUtil::get(templateId)->sum1D(compute, backupData.device(), elements);

  compute->sync();
  ProfileManager::Reset();
  {
    ProfileBlock("Reduce scan");
    for (uint i = 0; i < iterations; i++)
    {
      ComputeUtil::get(templateId)->sum1D(compute, backupData.device(), elements, true);
    }
  };

  compute->sync();

  float mean = ProfileManager::Get_Time_Since_Reset() / iterations;
  printStats(mean, elements, 1, sizeof(DataType));

  ComputeUtil::get(templateId)->sum1D(compute, data.device(), elements, true);
  data.syncHost(0, 1);
  compute->sync();

  std::cout << sum << " " << data.host()->at(0) << "\n";
  assert(abs(sum - data.host()->at(0)) <= .001f);

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
    particle.position = Real3(1);
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
  utilSetting[ComputeUtilStructMemberType] = "float3";

  uint templateId = ComputeUtil::create(compute, utilSetting, &includes);

  ComputeUtil::get(templateId)->sumRegular2D(compute, particles.device(), elements, width, true);

  particles.syncHost();
  compute->sync();

  for (int i = 0; i < parts; i++)
  {
    if (abs(means[i][0] - particlesHost[i * width].position.x) > .00001f
      || abs(means[i][1] - particlesHost[i * width].position.y) > .00001f
      || abs(means[i][2] - particlesHost[i * width].position.z) > .00001f)
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
    particle.position = Real3(1);
    if (sectionIndex < (partitionsHost.size() - 1) && i == partitionsHost[sectionIndex + 1].offset)
    {
      means.push_back(sum / float(partitionsHost[sectionIndex + 1].offset - partitionsHost[sectionIndex].offset));
      sum = 0;
      sectionIndex++;
    }

    IdentityInfo particleIdentity;
    resetIdentity(particleIdentity);
    setInstanceId(particleIdentity, sectionIndex);
    particle.identity = particleIdentity;

    particlesHost.push_back(particle);
    sum += particle.position;
  }

  means.push_back(sum / float(elements - partitionsHost[sectionIndex].offset));

  particles.syncDevice();
  partitions.syncDevice();

  vector<string> includes = { "ParticleStruct.h" };
  map<ComputeUtilKey, string> utilSetting;

  utilSetting[ComputeUtilStructType] = "ParticleStruct";
  utilSetting[ComputeUtilStructMember] = "position";
  utilSetting[ComputeUtilStructMemberType] = "float3";
  utilSetting[ComputeUtilIdentityFunction] = "getInstanceId";
  utilSetting[ComputeUtilIdentityStructType] = "ParticleStruct";
  utilSetting[ComputeUtilIdentityStructMember] = "identity";

  uint templateId = ComputeUtil::create(compute, utilSetting, &includes);

  ComputeUtil::get(templateId)->sumIrregular2D(compute, particles.device(), particles.device(), partitions.device(), partitionCount.device(), elements, width, true);

  particles.syncHost();
  compute->sync();

  for (uint i = 0; i < partitionsHost.size(); i++)
  {
    if (abs(means[i][0] - particlesHost[partitionsHost[i].offset].position.x) > .00001f
      || abs(means[i][1] - particlesHost[partitionsHost[i].offset].position.y) > .00001f
      || abs(means[i][2] - particlesHost[partitionsHost[i].offset].position.z) > .00001f)
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
    if (abs(means[i][0] - particlesConsolidated.host()->at(i).position.x) > .00001f
      || abs(means[i][1] - particlesConsolidated.host()->at(i).position.y) > .00001f
      || abs(means[i][2] - particlesConsolidated.host()->at(i).position.z) > .00001f)
    {
      std::cout << i << " " << means[i] << " " << particlesConsolidated.host()->at(i).position << "\n";
      assert(0);
    }
  }
  printf("Consolidation test passed!\n");
}

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

template<class DataType> void test1DPrefixScan(ComputeInterface* compute)
{
  printf("\nTesting 1D prefix scan:\n");

  DeviceArray<DataType> data(compute, NULL, true);
  DeviceArray<DataType> backupData(compute, NULL, false);
  vector<DataType> prefixSum;

  const int elements = 12345678;// 1024 * 1024 * 16;
  DataType sum = 0;
  const uint iterations = 20;

  data.host()->reserve(elements);
  for (uint i = 0; i < elements; i++)
  {
    data.host()->push_back(rand());
    prefixSum.push_back(sum);
    sum += data.host()->at(i);
  }

  backupData.resize(elements, false);
  data.syncDevice();
  compute->copyBuffer(data.device(), backupData.device(), 0, 0, elements * sizeof(DataType));

  map<ComputeUtilKey, string> utilSetting;
  utilSetting[ComputeUtilStructType] = "uint";
  utilSetting[ComputeUtilStructTypeIntegral] = "1";
  uint templateId = ComputeUtil::create(compute, utilSetting, NULL);
  ComputeUtil::get(templateId)->prefixScan1D(compute, backupData.device(), backupData.device(), elements);

  compute->sync();
  ProfileManager::Reset();
  {
    ProfileBlock("Prefix scan");
    for (uint i = 0; i < iterations; i++)
    {
      ComputeUtil::get(templateId)->prefixScan1D(compute, backupData.device(), backupData.device(), elements);
    }
  }
  compute->sync();

  float mean = ProfileManager::Get_Time_Since_Reset() / iterations;

  printStats(mean, elements, 2, sizeof(uint));

  ComputeUtil::get(templateId)->prefixScan1D(compute, data.device(), data.device(), elements);
  data.syncHost();
  compute->sync();

  for (uint i = 0; i < prefixSum.size(); i++)
  {
    if (abs((float)prefixSum[i] - data.host()->at(i)) > .00001f)
    {
      std::cout << i << " " << prefixSum[i] << " " << data.host()->at(i) << "\n";
      assert(0);
    }
  }

  printf("1D prefix scan test passed!\n");
}

template<class DataType> void test1DCompaction(ComputeInterface* compute)
{
  printf("\nTesting 1D compaction pass:\n");

  DeviceArray<DataType> count(compute, NULL, true);
  DeviceArray<DataType> compactIndexArray(compute, NULL, true);
  DeviceArray<DataType> selectionArray(compute, NULL, true);
  vector<uint> statusOutput;

  const int elements = 12345678;// 1024 * 1024 * 16;
  DataType sum = 0;
  const uint iterations = 20;

  selectionArray.host()->reserve(elements);
  statusOutput.reserve(elements);
  for (uint i = 0; i < elements; i++)
  {
    int value = rand() & 1;
    selectionArray.host()->push_back(value);
    if (value)
    {
      statusOutput.push_back(i);
    }
  }

  count.resize(1, false);
  selectionArray.syncDevice();
  compactIndexArray.resize(elements, false);

  map<ComputeUtilKey, string> utilSetting;
  utilSetting[ComputeUtilStructType] = "uint";
  utilSetting[ComputeUtilStructTypeIntegral] = "1";
  uint templateId = ComputeUtil::create(compute, utilSetting, NULL);
  ComputeUtil::get(templateId)->compactSparseArray(compute, count.device(), compactIndexArray.device(), selectionArray.device(), elements);

  compute->sync();

  ProfileManager::Reset();
  {
    ProfileBlock("Compact sparse array");
    for (uint i = 0; i < iterations; i++)
    {
      ComputeUtil::get(templateId)->compactSparseArray(compute, count.device(), compactIndexArray.device(), selectionArray.device(), elements);
    }
  }
  compute->sync();

  float mean = ProfileManager::Get_Time_Since_Reset() / iterations;

  printStats(mean, elements, 2, sizeof(uint));

  ComputeUtil::get(templateId)->compactSparseArray(compute, count.device(), compactIndexArray.device(), selectionArray.device(), elements);
  count.syncHost();
  compactIndexArray.syncHost();
  compute->sync();

  assert(count.host()->at(0) == statusOutput.size());

  for (uint i = 0; i < statusOutput.size(); i++)
  {
    if (statusOutput[i] != compactIndexArray.host()->at(i))
    {
      std::cout << i << " " << statusOutput[i] << " " << compactIndexArray.host()->at(i) << "\n";
      assert(0);
    }
  }

  printf("Compact 1D sparse array test passed!\n");
}

bool sortFunction(SortNode32 i, SortNode32 j)
{
  return (i.key < j.key);
}

/*void test1DBitonicSort32Bit(ComputeInterface* compute)
{
printf("\nTesting 1D bitonic sort:\n");

DeviceArray<SortNode32> data(compute, NULL, true);
vector<uint> prefixSum;
const int elements = 1024;

data.host()->reserve(elements);
for (int i = 0; i < elements; i++)
{
data.host()->push_back(SortNode32());
data.host()->back().key = rand();
data.host()->back().value = i;
prefixSum.push_back(0);
}

data.syncDevice();

map<ComputeUtilKey, string> utilSetting;
utilSetting[ComputeUtilStructType] = "uint";
uint templateId = ComputeUtil::create(compute, utilSetting, NULL);

ComputeUtil::get(templateId)->bitonicSort32Bit(compute, data.device(), elements);

data.syncHost();
compute->sync();

for (uint i = 0; i < prefixSum.size(); i++)
{
//if (abs((float)prefixSum[i] - data.host()->at(i)) > .00001f)
{
//std::cout << i << " " << prefixSum[i] << " " << data.host()->at(i) << "\n";
std::cout << i << " " << data.host()->at(i).key << " " << data.host()->at(i).value << "\n";
//assert(0);
}
}

printf("1D bitonic sort test passed!\n");
}*/

void test1DRadixSort32Bit(ComputeInterface* compute)
{
  printf("\nTesting 1D radix sort:\n");

  DeviceArray<SortNode32> destination(compute, NULL, true);
  DeviceArray<SortNode32> data(compute, NULL, true);
  vector<SortNode32> sortedData;
  const int elements = 12345678;// 1024 * 1024 * 4;

  data.host()->reserve(elements);
  destination.resize(elements, false);

  for (int i = 0; i < elements; i++)
  {
    data.host()->push_back(SortNode32());
    data.host()->back().key = rand() | (rand() << 16);
    data.host()->back().value = i;

    sortedData.push_back(data.host()->at(i));
  }

  data.syncDevice();
  compute->sync();

  map<ComputeUtilKey, string> utilSetting;
  utilSetting[ComputeUtilStructType] = "uint";
  utilSetting[ComputeUtilStructTypeIntegral] = "1";
  uint templateId = ComputeUtil::create(compute, utilSetting, NULL);

  ComputeUtil::get(templateId)->radixSort32Bit(compute, destination.device(), data.device(), elements);
  compute->sync();

  uint iterations = 1;
  float cumulativeTime = 0;
  for (uint i = 0; i < iterations; i++)
  {
    data.syncDevice();
    compute->sync();

    ProfileManager::Reset();
    {
      ProfileBlock("Radix sort");
      ComputeUtil::get(templateId)->radixSort32Bit(compute, destination.device(), data.device(), elements);
      compute->sync();
    }
    cumulativeTime += ProfileManager::Get_Time_Since_Reset();
    ProfileManager::dumpAll(stdout);
    ProfileManager::Increment_Frame_Counter();
  }

  float mean = cumulativeTime / iterations;
  printf("Average: %f\n", mean);
  printf("Elts/sec: %f\n", elements * 1000.f / mean);
  printf("Bandwidth util: %f\n", 4 * 2 * (elements * 1000.f / mean) * (3 * 32 / 4) / float(1024 * 1024 * 1024));
  destination.syncHost();
  compute->sync();

  std::sort(sortedData.begin(), sortedData.end(), sortFunction);

  for (uint i = 0; i < sortedData.size(); i++)
  {
    if (abs((float)sortedData[i].key - destination.host()->at(i).key) > .00001f)
    {
      std::cout << i << " " << sortedData.at(i).key << " " << sortedData.at(i).value << " " <<
        destination.host()->at(i).key << " " << destination.host()->at(i).value << "\n";
      assert(0);
    }
  }

  printf("1D radix sort test passed!\n");
}

int main(int argc, char** argv)
{
  compute = new ComputeInterface();
  compute->create();

  testBandwidthRW(compute);
  //testSectionOffsets(compute);
  //testBandwidthRead(compute);
  test1DMean<float>(compute);
  testRegular2DMean(compute);
  testIrregular2DMean(compute);
  test1DPrefixScan<uint>(compute);
  test1DCompaction<uint>(compute);
  //test1DBitonicSort32Bit(compute);
  test1DRadixSort32Bit(compute);
  //testEquation(compute);

  return 0;
}
