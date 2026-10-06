#include "lua_dual_arm.h"

#include <map>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include <le3/le3.h>
#include <scripting/le3_script_bindings.h>

#include "dabfc/dual_arm_simulator.h"
#include "dabfc/ur5e.h"

using namespace le3;
using namespace dabfc;

namespace {

DualArmSimulator& sim()
{
    static DualArmSimulator simulator = [] {
        const std::string meshDir =
            LE3GetConfig<std::string>("LE3GameConfig.ProjectPath") + "/ur5e/visual";
        const std::shared_ptr<const RobotArm> ur5e = std::make_shared<UR5e>(meshDir);
        return DualArmSimulator(ur5e, SE3{}, ur5e, SE3{});
    }();
    return simulator;
}

std::map<std::string, int>& attachedArms() // UR5e script object name -> arm index (0 or 1)
{
    static std::map<std::string, int> arms;
    return arms;
}

int armIndex(lua_State* L, int idx)
{
    const int i = static_cast<int>(luaL_checkinteger(L, idx)) - 1;
    if (i < 0 || i > 1) luaL_error(L, "DualArm: arm index must be 1 or 2");
    return i;
}

// Joint values of arm i from the table at idx
Eigen::VectorXd jointsFrom(lua_State* L, int idx, int i)
{
    luaL_checktype(L, idx, LUA_TTABLE);
    Eigen::VectorXd q(static_cast<Eigen::Index>(lua_rawlen(L, idx)));
    if (q.size() != sim().arm(i)->dof())
        luaL_error(L, "DualArm: expected %d joint values", sim().arm(i)->dof());
    for (int k = 0; k < q.size(); ++k) {
        lua_rawgeti(L, idx, k + 1);
        q[k] = lua_tonumber(L, -1);
        lua_pop(L, 1);
    }
    return q;
}

void pushNumbers(lua_State* L, const std::vector<double>& values)
{
    lua_createtable(L, static_cast<int>(values.size()), 0);
    for (int k = 0; k < std::ssize(values); ++k) {
        lua_pushnumber(L, values[k]);
        lua_rawseti(L, -2, k + 1);
    }
}

// Appends px, py, pz, qw, qx, qy, qz
void appendSE3(std::vector<double>& out, const SE3& pose)
{
    const Eigen::Vector3d p = pose.translation();
    const Eigen::Quaterniond r = pose.rotation();
    out.insert(out.end(), {p.x(), p.y(), p.z(), r.w(), r.x(), r.y(), r.z()});
}

} // namespace

// (name, px, py, pz, qw, qx, qy, qz) -> arm index; the UR5e base frame is the object transform
FBIND(DualArm, attach)
    GET_STRING(name)
    GET_VEC3_(p)
    GET_NUMBER(qw)
    GET_NUMBER(qx)
    GET_NUMBER(qy)
    GET_NUMBER(qz)
    std::map<std::string, int>& arms = attachedArms();
    const auto it = arms.find(name);
    const int i = it != arms.end() ? it->second : static_cast<int>(arms.size());
    if (i > 1) luaL_error(L, "DualArm: cannot attach a third arm (%s)", name.c_str());
    arms[name] = i;
    const SE3 zUpToYUp(Eigen::Vector3d::Zero(),
                       {-std::numbers::pi / 2, 0, 0}); // URDF is Z-up, LE3 is Y-up
    sim().setBase(i, SE3({p.x, p.y, p.z}, Eigen::Quaterniond(qw, qx, qy, qz)) * zUpToYUp);
    PUSH_NUMBER(i + 1)
FEND()

// (i) -> {{name = , mesh = , parts = {...}}, ...}: the links that have a mesh (as in mesh_poses)
FBIND(DualArm, links)
    const int i = armIndex(L, idx++);
    lua_newtable(L);
    const std::shared_ptr<const RobotArm> arm = sim().arm(i);
    int n = 0;
    for (const Link& link : arm->links()) {
        if (link.mesh.empty()) continue;
        lua_newtable(L);
        lua_pushstring(L, link.name.c_str());
        lua_setfield(L, -2, "name");
        lua_pushstring(L, link.mesh.c_str());
        lua_setfield(L, -2, "mesh");
        LE3GetScriptSystem().pushStringArray(link.parts);
        lua_setfield(L, -2, "parts");
        lua_rawseti(L, -2, ++n);
    }
    rcount = 1;
FEND()

// (i) -> {px, py, pz, qw, qx, qy, qz, ...}: world pose of every mesh of links(i), current state
FBIND(DualArm, mesh_poses)
    const int i = armIndex(L, idx++);
    const std::shared_ptr<const RobotArm> arm = sim().arm(i);
    const std::vector<SE3> poses = arm->meshPoses(sim().state()[i], sim().base(i));
    std::vector<double> out;
    for (int k = 0; k < std::ssize(poses); ++k)
        if (!arm->links()[k].mesh.empty()) appendSE3(out, poses[k]);
    pushNumbers(L, out);
    rcount = 1;
FEND()

// (i) -> px, py, pz, qw, qx, qy, qz: world pose of the tool (flange) frame in the current state
FBIND(DualArm, tool_pose)
    const int i = armIndex(L, idx++);
    const std::shared_ptr<const RobotArm> arm = sim().arm(i);
    std::vector<double> out;
    appendSE3(out, arm->fk(sim().state()[i], sim().base(i))[arm->toolLink()]);
    for (double v : out) {
        PUSH_NUMBER(v)
    }
FEND()

// (i) -> {q1, ..., qn}
FBIND(DualArm, get_joints)
    const Eigen::VectorXd& q = sim().state()[armIndex(L, idx++)];
    pushNumbers(L, std::vector<double>(q.begin(), q.end()));
    rcount = 1;
FEND()

// (i, {q1, ..., qn})
FBIND(DualArm, set_joints)
    const int i = armIndex(L, idx++);
    DualArmSimulator::State q = sim().state();
    q[i] = jointsFrom(L, idx++, i);
    sim().setState(q);
FEND()

// (sx, sy, sz, px, py, pz, qw, qx, qy, qz): static box obstacle with side lengths s, centred at p
FBIND(DualArm, add_box)
    GET_VEC3_(s)
    GET_VEC3_(p)
    GET_NUMBER(qw)
    GET_NUMBER(qx)
    GET_NUMBER(qy)
    GET_NUMBER(qz)
    sim().addObstacle(std::make_shared<fcl::Boxd>(s.x, s.y, s.z),
                      SE3({p.x, p.y, p.z}, Eigen::Quaterniond(qw, qx, qy, qz)));
FEND()

// ([q1, q2]) -> bool: is the state (default: the current one) collision free? Leaves it unchanged
FBIND(DualArm, is_collision_free)
    const bool free = lua_gettop(L) == 0
                          ? sim().isCollisionFreeNow()
                          : sim().isCollisionFreeAt({jointsFrom(L, 1, 0), jointsFrom(L, 2, 1)});
    PUSH_BOOL(free)
FEND()

LIB(DualArm, attach, links, mesh_poses, tool_pose, get_joints, set_joints, add_box,
    is_collision_free)

void registerDualArmBindings()
{
    lua_State* L = LE3GetScriptSystem().getLuaState();
    REGISTER(DualArm);
}
