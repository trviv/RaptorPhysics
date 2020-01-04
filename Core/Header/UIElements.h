#ifndef UIELEMENTS_H
#define UIELEMENTS_H

#include "GLClass.h"

enum UIElementType
{
  UI_ELEMENT_BOOL,
  UI_ELEMENT_STRING
};

class UIElement
{
  UIElementType type;
  Texture texture;
  void*   font;

public:
  string  identifier;
  bool    boolValue;
  string  stringValue;

  UIElement(const string& identifier, const bool value, const char* iconFont = NULL, ushort iconId = 0, const char* font = NULL);

  UIElement(const string& identifier, const char* value, const char* iconFont = NULL, ushort iconId = 0, const char* font = NULL);

  void render(uint width, uint height);
};

#endif
