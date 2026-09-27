#include <le3/le3.h>
#include <le3/ui/_NMB.h>
using namespace le3;

#include <filesystem>

#include <fmt/core.h>
#include <fmt/format.h>

class SDABFC_Visualization : public LE3GameLogic {

};

int main() {
    LE3Application app(std::make_unique<SDABFC_Visualization>());
    app.run();
    return 0;
}