#include "UnifiedPhysics.h"

static ComputeInterface* compute;

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
  DeviceArray<SectionData>    partitions(compute, NULL, true);

  uint width = 83;
  const uint parts = 625;
  const int elements = width * parts;
  Real3 sum = 0;

  for (int i = 0; i < parts; i++)
  {
    SectionData section;
    section.offsets[SECTION_DATA_NODE] = i * width;
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
    //particle.identity = sectionIndex;
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
  DeviceArray<SectionData>    partitions(compute, NULL, true);

  uint width = 1;
  const uint parts = 1022;
  const int elements = (parts * (parts + 1)) >> 1;// width * parts;
  Real3 sum = 0;

  for (int i = 0; i < parts; i++)
  {
    SectionData section;
    section.offsets[SECTION_DATA_NODE] = i ? partitions.host()->at(i - 1).offsets[SECTION_DATA_NODE] + width : 0;
    section.identity.setIdentity(1, 0);
    partitions.host()->push_back(section);
    width++;
  }

  particles.host()->reserve(elements);

  uint sectionIndex = 0;
  vector<SectionData> &partitionsHost = *partitions.host();
  vector<ParticleStruct> &particlesHost = *particles.host();
  vector<Real3> means;

  for (uint i = 0; i < elements; i++)
  {
    ParticleStruct particle;
    particle.position = 1;
    if (sectionIndex < (partitionsHost.size() - 1) &&
      i == partitionsHost[sectionIndex + 1].offsets[SECTION_DATA_NODE])
    {
      means.push_back(sum);
      sum = 0;
      sectionIndex++;
    }

    IdentityInfo particleIdentity;
    particleIdentity.setIdentity(0, sectionIndex);
    particleIdentities.host()->push_back(particleIdentity);

    particlesHost.push_back(particle);
    sum += particle.position;
  }

  means.push_back(sum);

  particles.syncDevice();
  partitions.syncDevice();
  particleIdentities.syncDevice();

  vector<string> includes = { "ParticleStruct.h" };
  map<ComputeUtilKey, string> utilSetting;
  utilSetting[ComputeUtilStructType] = "ParticleStruct";
  utilSetting[ComputeUtilStructMember] = "position";
  utilSetting[ComputeUtilIndexStructType] = "SectionData";
  utilSetting[ComputeUtilIndexStructMember] = "offsets[SECTION_DATA_NODE]";
  utilSetting[ComputeUtilIdentityFunction] = "getEntityId";
  utilSetting[ComputeUtilIdentityStructType] = "IdentityInfo";

  uint templateId = ComputeUtil::create(compute, utilSetting, &includes);

  ComputeUtil::get(templateId)->sumIrregular2D(compute, particles.device(), particleIdentities.device(), partitions.device(), elements, width, false);

  particles.syncHost();
  compute->sync();

  for (uint i = 0; i < partitionsHost.size(); i++)
  {
    if (means[i][0] != particlesHost[partitionsHost[i].offsets[SECTION_DATA_NODE]].position[0]
      || means[i][1] != particlesHost[partitionsHost[i].offsets[SECTION_DATA_NODE]].position[1]
      || means[i][2] != particlesHost[partitionsHost[i].offsets[SECTION_DATA_NODE]].position[2])
    {
      std::cout << i << " " << means[i] << " " << particlesHost[partitionsHost[i].offsets[SECTION_DATA_NODE]].position << "\n";
      assert(0);
    }
  }
  printf("Irregular 2D mean test passed!\n");

  printf("\nTesting section offsets count:\n");

  DeviceArray<ParticleStruct> particlesConsolidated(compute, NULL, true);
  DeviceArray<uint>           sectionOffsets(compute, NULL, true);
  DeviceArray<uint>           sectionOffsetCount(compute, NULL, true);

  particlesConsolidated.resize(parts, false);
  sectionOffsets.resize(parts, false);
  sectionOffsetCount.resize(1, false);

  ComputeUtil::get(templateId)->createSectionOffsets(compute,
    sectionOffsets.device(), sectionOffsetCount.device(), partitions.device(), parts);

  sectionOffsetCount.syncHost();
  compute->sync();

  if (sectionOffsetCount.host()->at(0) != parts)
  {
    printf("%d %d", sectionOffsetCount.host()->at(0), parts);
  }
  printf("Section offsets count test passed!\n");

  printf("\nTesting consolidation:\n");

  ComputeUtil::get(templateId)->copySectionOffsets(compute,
    particlesConsolidated.device(), particles.device(), sectionOffsets.device(), sectionOffsetCount.device(), parts);

  particlesConsolidated.syncHost();
  compute->sync();

  for (uint i = 0; i < particlesConsolidated.host()->size(); i++)
  {
    if (means[i][0] != particlesConsolidated.host()->at(i).position[0]
      || means[i][1] != particlesConsolidated.host()->at(i).position[1]
      || means[i][2] != particlesConsolidated.host()->at(i).position[2])
    {
      std::cout << i << " " << means[i] << " " << particlesConsolidated.host()->at(i).position << "\n";
      assert(0);
    }
  }
  printf("Consolidation test passed!\n");
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
}

void testSectionOffsets(ComputeInterface* compute)
{
  printf("\nTesting section offset:\n");

  vector<uint>              offsetOutput;
  DeviceArray<uint>         offsets(compute, NULL, true);
  DeviceArray<uint>         offsetCount(compute, NULL, true);
  DeviceArray<SectionData>  partitions(compute, NULL, true);

  const uint parts = 86878;
  uint nodes = 2;
  uint instanceCount = 1;
  uint totalNodes = 0;
  uint totalInstances = 0;

  for (int i = 0; i < parts; i++)
  {
    SectionData section;
    section.offsets[SECTION_DATA_NODE] = i ? (partitions.host()->at(i - 1).offsets[SECTION_DATA_NODE] +
      partitions.host()->at(i - 1).counts[SECTION_DATA_NODE]) : 0;
    section.counts[SECTION_DATA_NODE] = nodes * instanceCount;
    section.identity.setIdentity(instanceCount, 0);

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
  utilSetting[ComputeUtilIndexStructType] = "SectionData";
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
}

int main(int argc, char** argv)
{
  compute = new ComputeInterface();
  compute->create(1);

  testSectionOffsets(compute);
  test1DMean(compute);
  testRegular2DMean(compute);
  testIrregular2DMean(compute);
  //testPrefixSum1D(compute);
  //testEquation(compute);

  return 0;
}