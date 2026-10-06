-- UR5e arm with a Robotiq 2F-85 gripper. Kinematics, joint state and collision checks live in C++ (dabfc::UR5e and
-- dabfc::DualArmSimulator, exposed as the DualArm module, which exists only in viz); this script draws the arm.
-- Add in-editor as a ScriptObject with Classname = "UR5e"; the object's transform is the robot's base frame (Y-up).
-- Meshes are loaded from the resources/ur5e folder (packed by the editor into ur5e.dat): visual/*.stl, one file per URDF
-- material, baked from the .dae files by tools/dae2stl.py.
-- Everything created here is prefixed with __ENGINE__ so that it is never written into the saved scene or shared assets.
UR5e = LE3ScriptObject:new()

-- Editor material used for each material of the visual meshes. If a name does not exist in the project, a built-in
-- __ENGINE__ copy with the URDF colors is created instead, so the robot looks right without any editor setup
UR5e.materials = {LinkGrey = "M_ur5e_silver", JointGrey = "M_ur5e_silver", Black = "M_ur5e_silver", URBlue = "M_ur5e_blue",
                  RobotiqBlack = "M_robotiq_black", RobotiqGrey = "M_ur5e_silver"}
local URDF_COLORS = {M_ur5e_silver = {0.82, 0.82, 0.82, 1}, M_ur5e_blue = {0.49, 0.68, 0.80, 1}, M_robotiq_black = {0.22, 0.22, 0.22, 1}}

local function resolveMaterial(name) -- the editor material if it exists, otherwise the built-in copy
    if LE3AssetManager.has_material(name) then return name end
    if not LE3AssetManager.has_material("__ENGINE__" .. name) then
        LE3Material.load({Name = "__ENGINE__" .. name, ShaderName = "__ENGINE__S_default", DiffuseColor = URDF_COLORS[name] or {0.8, 0.8, 0.8, 1}, SpecularIntensity = 0.3, Shininess = 32})
    end
    return "__ENGINE__" .. name
end

function UR5e:init()
    if not DualArm then return end
    local px, py, pz = LE3Transform.get_position(self.transform)
    self.arm, self.parts = DualArm.attach(self.name, px, py, pz, LE3Transform.get_rotation(self.transform)), {}
    for k, L in ipairs(DualArm.links(self.arm)) do -- one static model per visual part, all parts of a link share its pose
        self.parts[k] = {}
        for _, part in ipairs(L.parts) do
            local file, name = L.mesh .. "_" .. part, "__ENGINE__" .. self.name .. "_" .. L.name .. "_" .. part
            local mesh = "__ENGINE__SM_ur5e_" .. file
            if not LE3AssetManager.has_static_mesh(mesh) then LE3StaticMesh.load({Name = mesh, Path = "/ur5e/visual/" .. file .. ".stl"}) end
            LE3StaticModel.load(self.scene, {Name = name, MeshName = mesh, MaterialName = resolveMaterial(UR5e.materials[part]), IsRigidBody = false})
            table.insert(self.parts[k], LE3Object.get_transform(LE3Scene.get_object(self.scene, name)))
        end
    end
    self:update(0)
end

-- Joint space (radians): {shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, gripper}; gripper: 0 = open (85 mm), 0.8 = closed
function UR5e:getJoints() return DualArm.get_joints(self.arm) end
function UR5e:setJoints(q) DualArm.set_joints(self.arm, q) end
function UR5e:getToolPose() -- tool0 (flange) in world: position {x,y,z}, rotation {w,x,y,z}
    local px, py, pz, qw, qx, qy, qz = DualArm.tool_pose(self.arm)
    return {px, py, pz}, {qw, qx, qy, qz}
end

function UR5e:update(deltaTime)
    if not self.arm then return end
    local P = DualArm.mesh_poses(self.arm)
    for k, parts in ipairs(self.parts) do
        local o = 7 * (k - 1)
        for _, t in ipairs(parts) do
            LE3Transform.set_position(t, P[o + 1], P[o + 2], P[o + 3])
            LE3Transform.set_rotation(t, P[o + 4], P[o + 5], P[o + 6], P[o + 7])
        end
    end
end
