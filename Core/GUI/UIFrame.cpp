#include "UIFrame.h"
#include <algorithm>

// Compact view options F065, F150,
// Expanded view option F066
UIFrame::UIFrame(const string &name, float width, float height)
  :shrink(false), name(name), compactText(" "+UIElement::getIconAsString("fa-solid-900", 0xF065)), compactOptionText(" "+UIElement::getIconAsString("fa-solid-900", 0xF066))
{
  size[0] = width;
  size[1] = height;
  addElement(new UIElement("frame-switch", compactText));
  ((UIElement*)uiElements[0])->autoWidth = true;
  ((UIElement*)uiElements[0])->setAlignment(Right|FloatX);
}

UIFrame::UIFrame(const string &name, float width, float height, float posX, float posY)
  :UIFrame(name, width, height)
{
  fixed = true;
  pos[0] = posX;
  pos[1] = posY;
}

UIFrame::UIFrame(const string &name, float width, float height, uint alignment)
  :UIFrame(name, width, height)
{
  this->alignment = alignment;
}

UIFrame::~UIFrame()
{
}

void UIFrame::addElement(UIElement* object)
{
  uiElements.push_back(object);
  uiElementMap[object->identifier] = (uint)uiElements.size() - 1;
}

UIElement* UIFrame::getElement(const string& name)
{
  return (UIElement*)uiElements[uiElementMap[name]];
}

bool UIFrame::render()
{
  bool changed = false;
  ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize;

  ImGui::Begin(name.c_str(), NULL, windowFlags);

  if (ImGui::GetCurrentContext()->LastActiveId == uiElements[0]->getUIID())
  {
    shrink = !shrink;
    changed = true;
  }

  // only set collapsed icon if defined, else leave it upto the instance
  if (compactText.size())
  {
    UIElement* element = (UIElement*)uiElements[0];
    // set collapsed if shrunk
    if (isShrunk() && compactText.size())
    {
      element->setText(compactText);
    }
    // if not collapsed and text changed by instance, then use it
    else if (element->getText() == compactText)
    {
      element->setText(compactOptionText);
    }
  }

  // display only first sub elements if shrunk
  changed |= (shrink ? uiElements[0]->render() : uiElements.render());

  if (text.size())
  {
    ImGui::SameLine();
    ImGui::Text("%s", text.c_str());
  }

  size[0] = ImGui::GetWindowSize().x;
  size[1] = ImGui::GetWindowSize().y;
  ImGui::SetWindowSize({size[0], size[1]});

  ImGui::SetWindowPos({pos[0], pos[1]});
  ImGui::End();
  return changed;
}

void UIFrame::setCompactText(const string& compactText)
{
  uiElements[0]->setText(" "+compactText);
  this->compactText = " "+compactText;
}

void UIFrame::setCompactOptionText(const string& compactOptionText)
{
  this->compactOptionText = compactOptionText;
}
