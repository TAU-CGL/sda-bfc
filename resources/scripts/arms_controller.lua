-- ImGui panel with joint and gripper sliders for two UR5e arms. Add as a ScriptObject with Classname = "ArmsController".
-- Dragging moves an arm live. On release, if the arms are in collision (self-collision, each other, an obstacle), it
-- bisects between that arm's last collision-free configuration and the released one and settles on the configuration
-- where the links just touch (to about 0.1 deg). Collision checks are exact and immediate (DualArm, C++).
ArmsController = LE3ScriptObject:new()
ArmsController.arms = {"r1", "r2"} -- names of the UR5e script objects in the scene
ArmsController.obstacles = {"floor"} -- scene boxes (BoxExtent {1,1,1}, sized by their scale) that the arms must not touch

local DEG = 180 / math.pi
local SLIDERS = { -- label, min, max (degrees for joints, radians for the gripper)
    {"shoulder_pan", -360, 360}, {"shoulder_lift", -360, 360}, {"elbow", -180, 180},
    {"wrist_1", -360, 360}, {"wrist_2", -360, 360}, {"wrist_3", -360, 360}, {"gripper", 0, 0.8},
}
local BISECT_STEPS = 10 -- release in collision: halve the bracket this many times (a 90 deg drag ends within 0.1 deg of contact)

local function midpoint(a, b) local m = {} for i = 1, #a do m[i] = 0.5 * (a[i] + b[i]) end return m end

function ArmsController:init()
    self.lastFree = {}
    if not DualArm then return end
    for _, name in ipairs(self.obstacles) do
        local obj = LE3Scene.get_object(self.scene, name)
        if obj then
            local t = LE3Object.get_transform(obj)
            local sx, sy, sz = LE3Transform.get_scale(t)
            local px, py, pz = LE3Transform.get_position(t)
            DualArm.add_box(sx, sy, sz, px, py, pz, LE3Transform.get_rotation(t))
        end
    end
end

-- Released in collision: bisect between arm i's last collision-free configuration (lo) and the released one (hi)
function ArmsController:settle(i)
    local q = {DualArm.get_joints(1), DualArm.get_joints(2)}
    local lo, hi = self.lastFree[i], q[i]
    if not lo or DualArm.is_collision_free(q[1], q[2]) then return end
    for _ = 1, BISECT_STEPS do
        q[i] = midpoint(lo, hi)
        if DualArm.is_collision_free(q[1], q[2]) then lo = q[i] else hi = q[i] end
    end
    DualArm.set_joints(i, lo)
end

function ArmsController:update(deltaTime)
    if not DualArm then return end
    local free = DualArm.is_collision_free()
    ImGui.SetNextWindowSize(440, 480)
    ImGui.Begin("Arms")
    ImGui.TextColored(free and 0.4 or 1, free and 1 or 0.4, 0.4, 1, free and "ok" or "COLLISION")
    ImGui.Separator()
    for _, name in ipairs(self.arms) do
        local arm = UR5e._refs[name]
        if not (arm and arm.arm) then
            ImGui.Text(name .. ": no UR5e object with this name")
        else
            local i, q = arm.arm, arm:getJoints()
            if free then self.lastFree[i] = {table.unpack(q)} end
            for k, sl in ipairs(SLIDERS) do
                local scale = k < 7 and DEG or 1
                local v, changed = ImGui.SliderFloat(name .. " " .. sl[1], q[k] * scale, sl[2], sl[3], k < 7 and "%.1f deg" or "%.2f rad")
                if changed then q[k] = v / scale; arm:setJoints(q) end
                if ImGui.IsItemDeactivatedAfterEdit() then self:settle(i) end
            end
        end
        ImGui.Separator()
    end
    ImGui.End()
end
