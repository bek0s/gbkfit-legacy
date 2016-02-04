
#include "application.hpp"

int main(int argc, char** argv)
{
    gbkfit_app_cli::Application* app = new gbkfit_app_cli::Application();

    if (app->process_program_options(argc, argv))
    {
        if (app->initialize())
        {
            app->run();
        }
        app->shutdown();
    }

    delete app;

    app = nullptr;

    return EXIT_SUCCESS;
}
