@echo off
REM Publish the current main branch to a Hugging Face Space.
REM Usage:  deploy\deploy_hf.bat YOUR_HF_USERNAME
REM Needs: a clean git working tree on main, and a Space named gen-ai-video-analyzer (SDK: Docker).
REM When git asks for a password, paste a Hugging Face access token with write permission.
if "%1"=="" (echo Usage: deploy\deploy_hf.bat YOUR_HF_USERNAME & exit /b 1)
git checkout -B hf || exit /b 1
git rm -r -q docs
copy /y deploy\hf_README.md README.md >nul
git add README.md
git commit -m "Hugging Face Space build"
git push https://huggingface.co/spaces/%1/gen-ai-video-analyzer hf:main --force
git checkout main
