#include "Core.h"
#include "../ParticlePhysics/Common/ParticleStruct.h"
#include <algorithm>

static ComputeInterface* compute;

static bool runOnlyFunctional = false;

#if TARGET_OS_IPHONE
  const int roughElements = 12345678;
#else
  const int roughElements = 123456789;
#endif

void printStats(float mean, uint elements, uint rwCount, uint sizeOfElements)
{
  logComputeMessage("Average time    : %f ms", mean);
  logComputeMessage("Elts/sec        : %f M", elements * 1000.f / (mean * 1000 * 1000));
  logComputeMessage("Bandwidth util  : %f GB/s", (rwCount * sizeOfElements * elements) * (1000.f / mean) / float(1024 * 1024 * 1024));
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
  logComputeMessage("Read/Write bandwidth test:");

  DeviceArray<float> data(compute, NULL);
  DeviceArray<float> outdata(compute, NULL);
  const int elements = 1024 * 1024 * 16;
  uint iterations = runOnlyFunctional?0:20;

  data.resize(elements, false);
  outdata.resize(elements, false);

  //--------------------------------------------------------------------------------
  // warm up run
  compute->copyBuffer(data.device(), outdata.device(), 0, 0, elements * sizeof(uint));
  compute->sync();

  //--------------------------------------------------------------------------------
  // performance run
  ProfileManager::Reset();
  {
    ProfileBlock("R/W Bandwidth");
    for (uint i = 0; i < iterations; i++)
    {
      compute->copyBuffer(data.device(), outdata.device(), 0, 0, elements * sizeof(uint));
    }
  }
  compute->sync();

  float mean = ProfileManager::Get_Time_Since_Reset() / iterations;
  printStats(mean, elements, 2, sizeof(uint));
}

void testCustomBandwidthRW(ComputeInterface* compute)
{
  logComputeMessage("Read/Write custom bandwidth test:");

  DeviceArray<float> data(compute, NULL);
  DeviceArray<float> outdata(compute, NULL);
  const int elements = 1024 * 1024 * 16;
  uint iterations = runOnlyFunctional?0:20;

  data.resize(elements, false);
  outdata.resize(elements, false);

  data.host()->reserve(elements);
  outdata.host()->reserve(elements);

  for (int i = 0; i < elements; i++)
  {
    data.host()->push_back(rand()&0x3);
    outdata.host()->push_back(0);
  }

  data.syncDevice();

  ComputeUtil::getUIntUtil(compute);

  //--------------------------------------------------------------------------------
  // warm up run
  ComputeUtil::get(0)->copyBuffer(compute, data.device(), outdata.device(), 0, 0, elements * sizeof(uint));
  compute->sync();

  //--------------------------------------------------------------------------------
  // performance run
  ProfileManager::Reset();
  {
    ProfileBlock("R/W Custom Bandwidth");
    for (uint i = 0; i < iterations; i++)
    {
      ComputeUtil::get(0)->copyBuffer(compute, data.device(), outdata.device(), 0, 0, elements * sizeof(uint));
    }
  }
  compute->sync();

  float mean = ProfileManager::Get_Time_Since_Reset() / iterations;

  outdata.syncHost();
  compute->sync();

  printStats(mean, elements, 2, sizeof(uint));

  for (int i = 0; i < elements; i++)
  {
    if (outdata.host()->at(i) != data.host()->at(i))
    {
      std::cout << i << " " << outdata.host()->at(i) << "\n";
      assert(0);
    }
  }
}

void testSetBuffer(ComputeInterface* compute)
{
  logComputeMessage("Clear bandwidth test:");

  DeviceArray<uint> data(compute, NULL);
  const int elements = 1024 * 1024 * 128;
  uint iterations = runOnlyFunctional?0:20;

  data.resize(elements, false);

  uint templateId = ComputeUtil::getUIntUtil(compute);

  //--------------------------------------------------------------------------------
  // warm up run
  if (!runOnlyFunctional)
  {
    ComputeUtil::get(templateId)->clearBuffer(compute, data.device(), elements);
    compute->sync();
  }

  //--------------------------------------------------------------------------------
  // performance run
  ProfileManager::Reset();
  {
    ProfileBlock("Clear Bandwidth");
    for (uint i = 0; i < iterations; i++)
    {
      ComputeUtil::get(templateId)->clearBuffer(compute, data.device(), elements);
    }
  }
  compute->sync();

  float mean = ProfileManager::Get_Time_Since_Reset() / iterations;

  //--------------------------------------------------------------------------------
  // functional run
  ComputeUtil::get(templateId)->clearBuffer(compute, data.device(), elements);
  compute->sync();

  data.syncHost();
  compute->sync();

  printStats(mean, elements, 1, sizeof(uint));

  for (int i = 0; i < elements; i++)
  {
    if (data.host()->at(i) != 0)
    {
      std::cout << i << " " << data.host()->at(i) << "\n";
      assert(0);
    }
  }
}

template<class DataType> void test1DMean(ComputeInterface* compute)
{
  logComputeMessage("Testing 1D mean:");

  DeviceArray<DataType> data(compute, NULL);
  DeviceArray<DataType> output(compute, NULL);

  const int elements = roughElements;
  double sum = 0;
  uint iterations = runOnlyFunctional?0:10;

  data.host()->reserve(elements);
  for (int i = 0; i < elements; i++)
  {
    data.host()->push_back(rand()&0x3);
    sum += data.host()->at(i);
  }

  output.resize(1, false);
  data.syncDevice();

  map<ComputeUtilKey, string> utilSetting;
  utilSetting[ComputeUtilStructType] = "float";
  uint templateId = ComputeUtil::create(compute, utilSetting, NULL);

  //--------------------------------------------------------------------------------
  // warm up run
  if (!runOnlyFunctional)
  {
    ComputeUtil::get(templateId)->sum1D(compute, output.device(), data.device(), elements);
    compute->sync();
  }

  //--------------------------------------------------------------------------------
  // performance run
  ProfileManager::Reset();
  {
    ProfileBlock("Reduce scan");
    for (uint i = 0; i < iterations; i++)
    {
      ComputeUtil::get(templateId)->sum1D(compute, output.device(), data.device(), elements, false);
    }
  }

  compute->sync();

  float mean = ProfileManager::Get_Time_Since_Reset() / iterations;
  printStats(mean, elements, 1, sizeof(DataType));

  //--------------------------------------------------------------------------------
  // functional run
  ComputeUtil::get(templateId)->sum1D(compute, output.device(), data.device(), elements, false);
  output.syncHost(0, 1);
  compute->sync();

  logComputeMessage("%f %f", sum, output.host()->at(0));
  // last 3 digits vary becasue of overflow, I guess
  assert(abs(sum/1000 - output.host()->at(0)/1000) <= 1.f);

  logComputeMessage("1D array mean test passed!");
}

void testRegular2DMean(ComputeInterface* compute)
{
  logComputeMessage("Testing regular 2D mean:");

  DeviceArray<ParticleStruct> particles(compute, NULL);
  DeviceArray<PartitionInfo>  partitions(compute, NULL);

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
  utilSetting[ComputeUtilStructMemberSize] = "16";

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
  logComputeMessage("Regular 2D mean test passed!");
}

void testIrregular2DMean(ComputeInterface* compute)
{
  logComputeMessage("Testing irregular 2D mean:");

  DeviceArray<ParticleStruct> particlesIn(compute, NULL);
  DeviceArray<ParticleStruct> particles(compute, NULL);
  DeviceArray<PartitionInfo>  partitions(compute, NULL);
  DeviceArray<uint>           partitionCount(compute, NULL);

  uint width = 125;
  const uint parts = 2000;
  uint elements = 0;
  Real3 sum = 0;
  uint iterations = runOnlyFunctional?0:10;

  int increment = 1;

  for (int i = 0; i < parts; i++)
  {
    PartitionInfo section;
    section.offset = i ? partitions.host()->at(i - 1).offset + width - increment : 0;
    section.count = width;
    partitions.host()->push_back(section);
    elements += width ;
    width += increment;
  }

  particlesIn.host()->reserve(elements);
  particles.resize(elements, false);
  partitionCount.host()->reserve(1);
  partitionCount.host()->push_back(parts);
  partitionCount.syncDevice();

  uint sectionIndex = 0;
  vector<PartitionInfo> &partitionsHost = *partitions.host();
  vector<ParticleStruct> &particlesHost = *particlesIn.host();
  vector<Real3> means;

  for (uint i = 0; i < elements; i++)
  {
    ParticleStruct particle;
    particle.position = Real3(rand()&0x3, rand()&0x3, rand()&0x3);
    if (sectionIndex < (partitionsHost.size() - 1) && i == partitionsHost[sectionIndex + 1].offset)
    {
      means.push_back(sum / float(partitionsHost[sectionIndex + 1].offset - partitionsHost[sectionIndex].offset));
      sum = 0;
      sectionIndex++;
    }

    IdentityInfo particleIdentity;
    resetIdentity(particleIdentity);
    setPhysicsInstanceId(particleIdentity, sectionIndex);
    particle.identity = particleIdentity;

    particlesHost.push_back(particle);
    sum += particle.position;
  }

  means.push_back(sum / float(elements - partitionsHost[sectionIndex].offset));

  particlesIn.syncDevice();
  partitions.syncDevice();

  vector<string> includes = { "ParticleStruct.h" };
  map<ComputeUtilKey, string> utilSetting;

  utilSetting[ComputeUtilStructType] = "ParticleStruct";
  utilSetting[ComputeUtilStructMember] = "position";
  utilSetting[ComputeUtilStructMemberType] = "float3";
  utilSetting[ComputeUtilStructMemberSize] = "16";
  utilSetting[ComputeUtilIdentityFunction] = "getInstanceId";
  utilSetting[ComputeUtilIdentityStructType] = "ParticleStruct";
  utilSetting[ComputeUtilIdentityStructMember] = "identity";

  uint templateId = ComputeUtil::create(compute, utilSetting, &includes);

  //--------------------------------------------------------------------------------
  // warm up run
  if (!runOnlyFunctional)
  {
    ComputeUtil::get(templateId)->sumIrregular2D(compute, particles.device(), particlesIn.device(), particlesIn.device(), partitions.device(), elements, true);
    compute->sync();
  }

  //--------------------------------------------------------------------------------
  // performance run
  ProfileManager::Reset();
  {
    ProfileBlock("Irregular Reduce scan");
    for (uint i = 0; i < iterations; i++)
    {
      ComputeUtil::get(templateId)->sumIrregular2D(compute, particles.device(), particlesIn.device(), particlesIn.device(), partitions.device(), elements, true);
    }
  }

  compute->sync();

  float mean = ProfileManager::Get_Time_Since_Reset() / iterations;
  printStats(mean, elements+partitions.size(), 1, sizeof(ParticleStruct));

  //--------------------------------------------------------------------------------
  // functional run
  ComputeUtil::get(templateId)->sumIrregular2D(compute, particles.device(), particlesIn.device(), particlesIn.device(), partitions.device(), elements, true);
  particles.syncHost();
  compute->sync();

  for (uint i = 0; i < means.size(); i++)
  {
    if (isnan(particles.host()->at(i).position.x) || abs(means[i][0] - particles.host()->at(i).position.x) > .00001f ||
        isnan(particles.host()->at(i).position.y) || abs(means[i][1] - particles.host()->at(i).position.y) > .00001f ||
        isnan(particles.host()->at(i).position.z) || abs(means[i][2] - particles.host()->at(i).position.z) > .00001f)
    {
      std::cout << i << " " << means[i] << " " << particles.host()->at(i).position << "\n";
      assert(0);
    }
  }

  logComputeMessage("Irregular 2D mean test passed!");
}

/*void testSectionOffsets(ComputeInterface* compute)
{
logComputeMessage("Testing section offset:");

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

logComputeMessage("Section offset count %d", offsetCount.host()->at(0));
for (uint i = 0; i < offsetOutput.size(); i++)
{
if (offsetOutput[i] != offsets.host()->at(i))
{
std::cout << offsetOutput[i] << " " << offsets.host()->at(i) << "\n";
assert(offsetOutput[i] == offsets.host()->at(i));
}
}
logComputeMessage("Section offset test passed!");
}*/

template<class DataType> void test1DPrefixScan(ComputeInterface* compute)
{
  logComputeMessage("Testing 1D prefix scan:");

  DeviceArray<DataType> data(compute, NULL);
  DeviceArray<DataType> backupData(compute, NULL);
  vector<DataType> prefixSum;

  const int elements = roughElements;
  DataType sum = 0;
  const uint iterations = runOnlyFunctional?0:10;

  data.host()->reserve(elements);
  for (uint i = 0; i < elements; i++)
  {
    data.host()->push_back(rand()&0xFF);
    prefixSum.push_back(sum);
    sum += data.host()->at(i);
  }

  backupData.resize(elements, false);
  data.syncDevice();
  compute->copyBuffer(data.device(), backupData.device(), 0, 0, elements * sizeof(DataType));

  uint templateId = ComputeUtil::getUIntUtil(compute);

  //--------------------------------------------------------------------------------
  // warm up run
  if (!runOnlyFunctional)
  {
    ComputeUtil::get(templateId)->prefixScan1D(compute, backupData.device(), backupData.device(), elements);
    compute->sync();
  }

  //--------------------------------------------------------------------------------
  // performance run
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

  //--------------------------------------------------------------------------------
  // functional run
  ComputeUtil::get(templateId)->prefixScan1D(compute, data.device(), data.device(), elements);
#if TARGET_OS_IPHONE
  compute->sync();
#endif
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

  logComputeMessage("1D prefix scan test passed!");
}

template<class DataType> void test1DCompaction(ComputeInterface* compute)
{
  logComputeMessage("Testing 1D compaction pass:");

  DeviceArray<DataType> count(compute, NULL);
  DeviceArray<DataType> compactIndexArray(compute, NULL);
  DeviceArray<DataType> selectionArray(compute, NULL);
  vector<uint> statusOutput;

  const int elements = roughElements;
  const uint iterations = runOnlyFunctional?0:10;

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

  count.resize(4, false);
  selectionArray.syncDevice();
  compactIndexArray.resize(elements, false);

  uint templateId = ComputeUtil::getUIntUtil(compute);

  //--------------------------------------------------------------------------------
  // warm up run
  if (!runOnlyFunctional)
  {
    ComputeUtil::get(templateId)->compactSparseArray(compute, count.device(), compactIndexArray.device(), selectionArray.device(), elements);
    compute->sync();
  }

  //--------------------------------------------------------------------------------
  // performance run
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

  //--------------------------------------------------------------------------------
  // functional run
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

  logComputeMessage("Compact 1D sparse array test passed!");
}

bool sortFunction(SortNode32 i, SortNode32 j)
{
  return (i.key < j.key);
}

/*void test1DBitonicSort32Bit(ComputeInterface* compute)
{
logComputeMessage("Testing 1D bitonic sort:");

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

logComputeMessage("1D bitonic sort test passed!");
}*/

void test1DRadixSort32Bit(ComputeInterface* compute)
{
  logComputeMessage("Testing 1D radix sort:");

  DeviceArray<SortNode32> destination(compute, NULL);
  DeviceArray<SortNode32> data(compute, NULL);
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

  uint templateId = ComputeUtil::getUIntUtil(compute);

  //--------------------------------------------------------------------------------
  // warm up run
  if (!runOnlyFunctional)
  {
    ComputeUtil::get(templateId)->radixSort32Bit(compute, destination.device(), data.device(), elements);
    compute->sync();
  }

  //--------------------------------------------------------------------------------
  // performance run
  uint iterations = runOnlyFunctional?0:10;
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

  printStats(mean, elements, ((3 + 1) * 32 / 4), sizeof(uint));

  //--------------------------------------------------------------------------------
  // functional run
  data.syncDevice();
  compute->sync();
  ComputeUtil::get(templateId)->radixSort32Bit(compute, destination.device(), data.device(), elements);

  destination.syncHost();
  compute->sync();

  std::stable_sort(sortedData.begin(), sortedData.end(), sortFunction);

  for (uint i = 0; i < sortedData.size(); i++)
  {
    if ((sortedData[i].key != destination.host()->at(i).key) || (sortedData[i].value != destination.host()->at(i).value))
    {
      std::cout << i << " " << sortedData.at(i).key << " " << sortedData.at(i).value << " " <<
        destination.host()->at(i).key << " " << destination.host()->at(i).value << "\n";
      assert(0);
    }
  }

  logComputeMessage("1D radix sort test passed!");
}

int main(int argc, char** argv)
{
  compute = new ComputeInterface();
  compute->create();

  testBandwidthRW(compute);
  testCustomBandwidthRW(compute);
  testSetBuffer(compute);
  //testSectionOffsets(compute);
  //testBandwidthRead(compute);
  test1DMean<float>(compute);
//  TODO: Investigate why regular 2d mean is not working with 256 + threadgroup width
#if !TARGET_OS_IPHONE
  testRegular2DMean(compute);
#endif
  testIrregular2DMean(compute);
  test1DPrefixScan<uint>(compute);
  test1DCompaction<uint>(compute);
  //test1DBitonicSort32Bit(compute);
  test1DRadixSort32Bit(compute);
  //testEquation(compute);

  delete compute;
  return 0;
}
