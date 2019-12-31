#ifndef UIELEMENTS_H
#define UIELEMENTS_H

#include "ImageIO.h"

enum UIElementType
{
  UI_ELEMENT_BOOL,
  UI_ELEMENT_STRING
};

struct UIElement
{
  UIElementType type;
  string        name;
  bool          boolValue;
  string        stringValue;

  UIElement(const string& name, const bool value)
  {
    type = UI_ELEMENT_BOOL;
    this->name = name;
    boolValue = value;
  }

  UIElement(const string& name, const string& value)
  {
    type = UI_ELEMENT_STRING;
    this->name = name;
    stringValue = value;
  }

  void render(uint width, uint height);
};

#endif
