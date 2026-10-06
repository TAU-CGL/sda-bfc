#include <memory>

#include <le3/le3.h>

#include "lua_dual_arm.h"

class Visualization : public le3::LE3GameBase {
public:
    void init() override
    {
        setDisplayFPS(true);
        LE3GameBase::init();
    }
    void registerBindings() override { registerDualArmBindings(); }
};

int main()
{
    le3::LE3Application app(std::make_unique<Visualization>());
    app.run();
}
