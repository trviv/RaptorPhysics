#include "MemoryManager.h"
#include <vector>

MemoryManager memoryManager;

MemoryManager::MemoryManager()
{
  allocatedMemory.clear();
}

MemoryManager::~MemoryManager()
{
  dealloc();
}

void* MemoryManager::alloc(const MemoryManaged* managed, uint sizeInBytes)
{
  void* memory = ::malloc(sizeInBytes);
  if (allocatedMemory.find(managed) == allocatedMemory.end())
  {
    allocatedMemory[managed] = 0;
  }
  allocatedMemory[managed] += 1;

  return memory;
}

void MemoryManager::dealloc()
{
  vector<const MemoryManaged*> deallocList;
  for (auto &i : allocatedMemory)
  {
    if (i.second == 0)
    {
      i.first->freeManaged();
      deallocList.push_back(i.first);
    }
    else
    {
      i.second = 0;
    }
  }

  for (auto i : deallocList)
  {
    allocatedMemory.erase(i);
  }
}

void MemoryManager::hit(const MemoryManaged* managed)
{
  allocatedMemory[managed] += 1;
}
