#ifndef UIFRAME_H
#define UIFRAME_H

#include "UIElements.h"
#include <unordered_map>

/*!
@class Class representing user interface frame.
*/
class UIFrame : public UIObject
{
protected:
  unordered_map<string, uint> uiElementMap;
  vector<UIElement> uiElements;

  bool    shrink;
  string  name;

  void render();

public:
  string  text;

  UIFrame(const string &name, float width, float height, float posX, float posY);

  UIFrame(const string &name, float width, float height, uint alignment);

  UIFrame(const UIFrame &ref);

  ~UIFrame();

  void addElement(const UIElement& option);

  UIElement& getElement(const string& name);

  bool isShrunk() const { return shrink;};
};

#endif
