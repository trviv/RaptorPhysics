#ifndef UIELEMENTS_H
#define UIELEMENTS_H

#include "UIList.h"

enum UIElementType
{
  UI_ELEMENT_BOOL,
  UI_ELEMENT_STRING
};

/*!
@class Class holding user interface item data.
*/
class UIElement : public UIObject
{
  UIElementType type;
  Texture texture;
  void*   font;

public:
  static float ButtonWidth;
  static float ButtonHeight;

  static string getIconAsString(const char* iconFont = NULL, ushort iconId = 0);

  string  identifier;
  bool    boolValue;
  string  displayText;

  UIElement(const string& identifier, const bool value, const char* iconFont = NULL, ushort iconId = 0, const char* font = NULL);

  UIElement(const string& identifier, const string& text, const char* iconFont = NULL, ushort iconId = 0, const char* font = NULL);

  ~UIElement();

  void render();
};

#endif
