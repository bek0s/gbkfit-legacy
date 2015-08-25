
#include "application.hpp"

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    gbkfit_app_cli::application* app = new gbkfit_app_cli::application();

    if (app->initialize())
    {
        app->run();
    }

    app->shutdown();

    delete app;

    app = nullptr;

    return EXIT_SUCCESS;
}
