/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

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
    if (i.second == 0 && !(i.first->persistant))
    {
      i.first->freeManaged();
      deallocList.push_back(i.first);
    }
    else
    {
      i.second--;
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
