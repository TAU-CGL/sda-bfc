// Extra ImGui Lua bindings for this project. LightEngine3 only ships its property-panel widgets (inputs, checkboxes...),
// so windows, sliders and release detection are added here with the engine's own binding macros. Exposed as `ImGuiEx`.
#include <le3/le3.h>
#include <le3/scripting/le3_script_bindings.h>
#include <le3/ui/_imgui.h>
using namespace le3;

FBIND(ImGuiEx, Begin) GET_STRING(name) PUSH_BOOL(ImGui::Begin(name.c_str())) FEND()
FBIND(ImGuiEx, End) ImGui::End(); FEND()
FBIND(ImGuiEx, SetNextWindowSize) GET_NUMBER(w) GET_NUMBER(h) ImGui::SetNextWindowSize(ImVec2((float)w, (float)h), ImGuiCond_FirstUseEver); FEND()
FBIND(ImGuiEx, Separator) ImGui::Separator(); FEND()
FBIND(ImGuiEx, SliderFloat) // (label, value, min, max, format) -> value, changed
    GET_STRING(label) GET_NUMBER(value) GET_NUMBER(minValue) GET_NUMBER(maxValue) GET_STRING(format)
    float v = (float)value;
    bool changed = ImGui::SliderFloat(label.c_str(), &v, (float)minValue, (float)maxValue, format.c_str());
    PUSH_NUMBER(v) PUSH_BOOL(changed)
FEND()
FBIND(ImGuiEx, IsItemActive) PUSH_BOOL(ImGui::IsItemActive()) FEND()
FBIND(ImGuiEx, IsItemDeactivatedAfterEdit) PUSH_BOOL(ImGui::IsItemDeactivatedAfterEdit()) FEND()

LIB(ImGuiEx, Begin, End, SetNextWindowSize, Separator, SliderFloat, IsItemActive, IsItemDeactivatedAfterEdit)

void registerImGuiEx() {
    lua_State* L = LE3GetScriptSystem().getLuaState();
    luaL_requiref(L, "ImGuiEx", le3::luaopen_ImGuiEx, 1);
    lua_pop(L, 1);
}
