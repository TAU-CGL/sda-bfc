#ifndef SDABFC_LUA_DUAL_ARM_H_
#define SDABFC_LUA_DUAL_ARM_H_

// Lua module DualArm: the scene's two UR5e arms as one dabfc::DualArmSimulator. Arm indices are
// 1 and 2, in the order the UR5e script objects attach; poses are in LE3 world coordinates (Y-up)
void registerDualArmBindings();

#endif // SDABFC_LUA_DUAL_ARM_H_
