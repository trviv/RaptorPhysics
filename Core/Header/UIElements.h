#ifndef UIELEMENTS_H
#define UIELEMENTS_H

#include "GLClass.h"

enum UIElementType
{
  UI_ELEMENT_BOOL,
  UI_ELEMENT_STRING
};

/*!
@class Class holding user interface item data.
*/
class UIElement
{
  UIElementType type;
  Texture texture;
  void*   font;

  string getIconAsString(const char* iconFont = NULL, ushort iconId = 0)const;

public:
  static float ButtonWidth;
  static float ButtonHeight;

  string  identifier;
  bool    boolValue;
  string  displayText;

  UIElement(const string& identifier, const bool value, const char* iconFont = NULL, ushort iconId = 0, const char* font = NULL);

  UIElement(const string& identifier, const string& text, const char* iconFont = NULL, ushort iconId = 0, const char* font = NULL);

  void render(uint width = 0, uint height = 0);
};

#endif
