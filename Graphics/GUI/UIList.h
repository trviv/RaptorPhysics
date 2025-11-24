/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef UILIST_H
#define UILIST_H

#include <Graphics/GLClass.h>

/*!
@class Base class for all UI objects containing common functionalities.
*/
class UIObject
{
  friend class UIList;

protected:
  bool  fixed;
  float pos[2];
  float size[2];
  uint  alignment;

  string  text;
  uint    uiID;

public:
  enum UIObjectAlign
  {
    Bottom = 0x1,
    Right  = 0x2,
    FloatX = 0x4,
    FloatY = 0x8
  };

  UIObject();

  virtual ~UIObject() {}

  virtual bool render() = 0;

  void setAlignment(const uint align) {alignment = align;}

  uint getUIID()const {return uiID;}

  void setText(const string& text) {this->text = text;}

  const string& getText()const {return text;}

  bool overlap(const UIObject* other)const;
};

/*!
@class Class representing a list of objects and and provide common functionality.
*/
class UIList : public vector<UIObject*>
{
  void alignByAxis(UIObject* object, int axis);
  
public:
  /*!@member Common padding applied to each object in the list. [X = 0, Y = 1][Left = Top = 0, Right = Bottom = 1]*/
  float cornerPadding[2][2];

  UIList();

  ~UIList();

  bool render();
};

#endif
