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
  string  name;
  bool    boolValue;
  string  stringValue;
  Texture texture;
  void*   font;

  UIElement(const string& name, const bool value, const char* font = NULL);

  UIElement(const string& name, const char* value, const char* font = NULL);

  void render(uint width, uint height);
};

#endif
