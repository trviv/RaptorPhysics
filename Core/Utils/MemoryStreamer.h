#ifndef MEMORY_STREAMER_H
#define MEMORY_STREAMER_H

#include "IOInterface.h"
#include "MemoryManager.h"
#include <deque>
#include <stack>

/*!
@class Class to allocate and recycle memory based on a fixed max size.
*/
class MemoryStreamer
{
  uint maxSize;
  uint lastOffset;
  uint allocatedSize;

  /*!@member Available freed memory.*/
  stack<vector<uint8_t>*> availableMemory;

  /*!@member Memory currently allocated.*/
  deque<vector<uint8_t>*> allocatedMemory;

  void deallocItem();

public:
  MemoryStreamer();

  ~MemoryStreamer();

  /*!@function Maximum memory streamer should use.*/
  void setMaxCapacity(uint sizeInBytes);

  /*!@function Insert a new entry.*/
  void addItem(uint sizeInBytes);

  /*!@function Copy and insert memory as a new entry.*/
  void addItem(const void* data, uint sizeInBytes);

  /*!@function Append data at the tail end of last item in the stream.*/
  void appendToLast(const void* data, uint sizeInBytes);

  uint size()const { return (uint)allocatedMemory.size();}

  const void* getItem(uint index, uint memoryOffset)const;
};

#endif
