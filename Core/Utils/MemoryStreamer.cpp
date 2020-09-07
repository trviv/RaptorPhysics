#include "MemoryStreamer.h"

MemoryStreamer::MemoryStreamer()
{
  maxSize = 0;
  lastOffset = 0;
  allocatedSize = 0;
  allocatedMemory.clear();
}

MemoryStreamer::~MemoryStreamer()
{
  while (availableMemory.size())
  {
    delete availableMemory.top();
    availableMemory.pop();
  }

  for (auto i : allocatedMemory)
  {
    delete i;
  }
}

void MemoryStreamer::deallocItem()
{
  // check if dealloc is valid
  if (allocatedMemory.size())
  {
    vector<uint8_t>* mem = allocatedMemory.front();
    allocatedSize -= mem->size();
    availableMemory.push(mem);
    allocatedMemory.pop_front();
  }
  else
  {
    logComputeError("Size Allocation insufficient!");
  }
}

void MemoryStreamer::setMaxCapacity(uint sizeInBytes)
{
  maxSize = sizeInBytes;
}

void MemoryStreamer::addItem(uint sizeInBytes)
{
  if (maxSize == 0) return;

  while ((allocatedSize + sizeInBytes) > maxSize)
  {
    deallocItem();
  }

  vector<uint8_t>* newMemArray = NULL;

  if (availableMemory.size())
  {
    newMemArray = availableMemory.top();
    availableMemory.pop();
  }
  else
  {
    newMemArray = new vector<uint8_t>(sizeInBytes);
  }

  lastOffset = 0;
  allocatedSize += sizeInBytes;
  if (newMemArray->max_size() < sizeInBytes)
  {
    newMemArray->reserve(sizeInBytes);
  }
  newMemArray->resize(sizeInBytes);
  allocatedMemory.push_back(newMemArray);
}

void MemoryStreamer::addItem(void* data, uint sizeInBytes)
{
  addItem(sizeInBytes);
  appendToLast(data, sizeInBytes);
}

void MemoryStreamer::appendToLast(void* data, uint sizeInBytes)
{
  if (maxSize == 0) return;

  if ((lastOffset + sizeInBytes) > (allocatedMemory.back()->size()*4))
  {
    logComputeError("Trying to fill more data then allowed!");
  }

  memcpy(&(*(allocatedMemory.back()))[lastOffset], data, sizeInBytes);
  lastOffset += sizeInBytes;
}

const void* MemoryStreamer::getItem(uint index, uint memoryOffset)const
{
  return &(*allocatedMemory[index])[memoryOffset];
}
