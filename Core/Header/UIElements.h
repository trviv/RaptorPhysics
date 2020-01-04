#ifndef UIELEMENTS_H
#define UIELEMENTS_H

#include "GLClass.h"

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
  uint    iconId;
  void*   iconFont;

  UIElement(const string& name, const bool value, const char* iconFont = NULL, ushort iconId = 0, const char* font = NULL);

  UIElement(const string& name, const char* value, const char* iconFont = NULL, ushort iconId = 0, const char* font = NULL);

  void render(uint width, uint height);
};

#endif
