/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef UIFRAME_H
#define UIFRAME_H

#include "UIList.h"
#include "UIElements.h"
#include <unordered_map>

/*!
@class Class representing user interface frame.
*/
class UIFrame : public UIObject
{
protected:
  unordered_map<string, uint> uiElementMap;
  UIList  uiElements;

  bool    shrink;
  string  name;
  string  compactText;
  string  compactOptionText;

  bool render();

  explicit UIFrame(const string &name, float width, float height);

public:
  UIFrame(const string &name, float width, float height, float posX, float posY);

  UIFrame(const string &name, float width, float height, uint alignment);

  ~UIFrame();

  void addElement(UIElement* option);

  UIElement* getElement(const string& name);

  UIList& getElements() { return uiElements;}

  bool isShrunk() const { return shrink;}

  void setCompactText(const string& compactText);

  void setCompactOptionText(const string& compactOptionText);
};

#endif
