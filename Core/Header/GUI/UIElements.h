#ifndef UIELEMENTS_H
#define UIELEMENTS_H

#include "UIList.h"

/*!
@class Class representing user interface button.
*/
class UIElement : public UIObject
{
public:
  enum UIElementType
  {
    Bool,
    String
  };

private:
  UIElementType type;
  Texture texture;
  void*   font;

public:
  static float ButtonWidth;
  static float ButtonHeight;

  static string getIconAsString(const char* iconFont = NULL, ushort iconId = 0);

  string  identifier;
  bool    boolValue;

  UIElement(const string& identifier, const bool value, const char* iconFont = NULL, ushort iconId = 0, const char* font = NULL);

  UIElement(const string& identifier, const string& text, const char* iconFont = NULL, ushort iconId = 0, const char* font = NULL);

  ~UIElement();

  bool render();
};

#endif
