#ifndef UILIST_H
#define UILIST_H

#include "../GLClass.h"

/*!
@class Base class for all UI objects containing common functionalities.
*/
class UIObject
{
  friend class UIList;

protected:
  bool  refresh;
  float pos[2];
  float size[2];
  uint  alignment;

public:
  enum UIObjectAlign
  {
    Bottom = 0x1,
    Right  = 0x2,
    FloatX = 0x4,
    FloatY = 0x8
  };

  string  text;

  virtual ~UIObject() {}

  virtual void render() = 0;
};

/*!
@class Class representing a list of objects and and provide common functionality.
*/
class UIList : public vector<UIObject*>
{
  void alignByAxis(UIObject* object, int axis);
  
public:
  ~UIList();

  void render();
};

#endif
