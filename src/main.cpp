#include <le3/le3.h>
#include <le3/ui/_NMB.h>
using namespace le3;

#include <filesystem>

#include <fmt/core.h>
#include <fmt/format.h>

void registerImGuiEx(); // lua_imgui_ext.cpp
void registerPhysicsEx(); // lua_physics_ext.cpp
void registerMeshCollide(); // lua_mesh_collide.cpp

class SDABFC_Visualization : public LE3GameLogic {
public:
    void init() override {
        loadBootstrapConfig();
        registerImGuiEx();
        registerPhysicsEx();
        registerMeshCollide();
        loadProjectArchives();
        loadInitialScene();
        initFPSDisplay();
    }

    void update(float deltaTime) override {
        // Same as LE3GetSceneManager().updateScenes(deltaTime), except that Bullet is stepped exactly once per frame.
        // The engine's own call steps at a fixed 60 Hz, so above 60 FPS collision detection only runs every few frames
        // and the contact reports lag the poses set by scripts; the robots' collision bookkeeping needs them in lockstep.
        LE3GetPhysicsManager().update(1.0f / 60.0f + 1e-5f);
        for (auto& [name, scene] : LE3GetSceneManager().getScenes()) if (!scene->isInspected()) scene->preUpdate();
        for (auto& [name, scene] : LE3GetSceneManager().getScenes()) scene->update(deltaTime);
        for (auto& [name, scene] : LE3GetSceneManager().getScenes()) if (!scene->isInspected()) scene->postUpdate();
        updateFPSDisplay(deltaTime);
    }

    void render() override {
        LE3GetActiveScene()->draw();
    }

    void renderDebug() {
    }

    void handleInput(LE3Input input) override {
    }

private:
    float m_fpsPrintTimer = 0.0f;
    LE3UITextObjectPtr m_fpsTextObject;

    void loadBootstrapConfig() {
        std::string bootstrapContent;
        try {
            bootstrapContent = LE3DatBuffer::loadFromSystem(LE3GetDataFilePath(BOOTSTRAP_FILE)).toString(); // The bootstrap file #define is defined in CMake!
        } catch (std::runtime_error e) {
            NMB::show("ERROR", "Could not read boostrap file! Closing...", NMB::Icon::ICON_ERROR);
            exit(-1);
        }
        LE3GetScriptSystem().doString(bootstrapContent);
    }

    void loadProjectArchives() {
        std::string projectPath = LE3GetConfig<std::string>("LE3GameConfig.ProjectPath");

        // Preload also project & demo archives
        std::string projectFilename = "le3proj.dat";
        LE3GetDatFileSystem().addArchive("le3proj", fmt::format("{}/{}", projectPath, projectFilename));
        LE3GetDatFileSystem().addArchive("demo", LE3GetDataFilePath("demo.dat"));

        // Add the rest of the archives that are available
        for (const auto& entry : std::filesystem::directory_iterator(projectPath)) {
            if (entry.path().extension() == ".dat" && entry.path().filename() != projectFilename) {
                LE3GetDatFileSystem().addArchive(entry.path().stem().string(), fmt::format("{}/{}", projectPath, entry.path().filename().string()));
            }
        }

        // Run all scripts (==load all class types)
        if (LE3GetDatFileSystem().archiveExists("scripts")) {
            for (std::string script : LE3GetDatFileSystem().getFilesFromDir("/scripts")) {
                if (!script.ends_with(".lua")) continue;
                fmt::print("\tSCRIPT:{}\n", script);
                LE3GetScriptSystem().doFile(script);
            }
        }

        LE3GetAssetManager().reloadAssets();
    }

    void loadInitialScene() {
        std::string initialSceneName = LE3GetConfig<std::string>("LE3GameConfig.InitialScene");
        LE3GetPhysicsManager().reset();
        LE3GetSceneManager().createScene("__main__", m_engineState, "");
        LE3GetSceneManager().getScene("__main__")->load("/le3proj/scenes/__shared__.lua");
        LE3GetSceneManager().getScene("__main__")->load(fmt::format("/le3proj/scenes/{}", initialSceneName));

        if (!LE3GetActiveScene()->getObject(LE3_PLAYERSTART_OBJECT_NAME)) {
            LE3GetScriptSystem().pushUserType<LE3Scene>(LE3GetActiveScene().get());
            LE3GetScriptSystem().setGlobal("_activeScene");
            LE3GetScriptSystem().doString(
                fmt::format("LE3PlayerStart.load(_activeScene, {{Name = \"{}\", Classname = \"{}\"}})",
                    LE3_PLAYERSTART_OBJECT_NAME,
                    LE3_PLAYERSTART_DEFAULT_CLASS
                ));
        }
    }

    void initFPSDisplay() {
        LE3GetAssetManager().addFont("F_default", "/engine/fonts/Tahoma.ttf", 32.0f);
        LE3GetActiveScene()->addUITextObject("fpsText", "F_default");
        m_fpsTextObject = LE3GetActiveScene()->getObject<LE3UITextObject>("fpsText");
        m_fpsTextObject->getTransform().setPosition(glm::vec3(-0.98f, 0.92f, 0.0f));
        m_fpsTextObject->getTransform().setScale(glm::vec3(0.06f, 0.06f, 1.f));
        m_fpsTextObject->setText("FPS: 0");
        m_fpsTextObject->setTextColor(glm::vec4(0.f, 0.9f, 0.1f, 1.f));
    }

    void updateFPSDisplay(float deltaTime) {
        m_fpsPrintTimer += deltaTime;
        if (m_fpsPrintTimer >= 0.25f) {
            m_fpsPrintTimer = 0.0f;
            if (m_fpsTextObject) m_fpsTextObject->setText(fmt::format("FPS: {:.0f}", m_engineState.getFPS()));
        }
    }
};

int main() {
    LE3Application app(std::make_unique<SDABFC_Visualization>());
    app.run();
    return 0;
}