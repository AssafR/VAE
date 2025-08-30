@echo off
echo ========================================
echo VAE Training Configuration Setup
echo ========================================
echo.

if exist config.yaml (
    echo config.yaml already exists!
    echo.
    echo To edit your configuration:
    echo 1. Open config.yaml in your text editor
    echo 2. Update the huggingface.username field
    echo 3. Update the huggingface.token field
    echo.
    echo Press any key to open config.yaml in Notepad...
    pause >nul
    notepad config.yaml
) else (
    echo Creating config.yaml from template...
    copy config.template.yaml config.yaml
    echo.
    echo config.yaml created successfully!
    echo.
    echo Now you need to:
    echo 1. Get your Hugging Face token from: https://huggingface.co/settings/tokens
    echo 2. Edit config.yaml with your username and token
    echo.
    echo Press any key to open config.yaml in Notepad...
    pause >nul
    notepad config.yaml
)

echo.
echo Configuration setup complete!
echo You can now run: uv run VAE.py
pause
