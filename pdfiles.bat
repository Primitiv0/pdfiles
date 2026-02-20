@echo off
setlocal enabledelayedexpansion

:: PDfiles — Windows management script
:: Requires Docker Desktop

set "COMMAND=%~1"
if "%COMMAND%"=="" set "COMMAND=help"

if "%COMMAND%"=="deploy" goto :deploy
if "%COMMAND%"=="up" goto :up
if "%COMMAND%"=="update" goto :update
if "%COMMAND%"=="down" goto :down
if "%COMMAND%"=="logs" goto :logs
if "%COMMAND%"=="status" goto :status
if "%COMMAND%"=="help" goto :help
if "%COMMAND%"=="--help" goto :help
if "%COMMAND%"=="-h" goto :help

echo Unknown command: %COMMAND%
echo Run 'pdfiles.bat help' for usage.
exit /b 1

:deploy
set "DATA_PATH=%~2"
if "%DATA_PATH%"=="" (
    echo Usage: pdfiles.bat deploy DATA_PATH
    exit /b 1
)
if not exist "%DATA_PATH%" (
    echo ERROR: Data directory not found: %DATA_PATH%
    exit /b 1
)
if not exist .env (
    echo Creating .env...
    echo DATA_PATH=%DATA_PATH%> .env
    echo WEB_PORT=80>> .env
)
echo === PDfiles Deploy ===
echo Data path: %DATA_PATH%
echo.
echo Building containers...
docker compose build
echo.
echo Starting services...
docker compose up -d
echo.
echo Frontend: http://localhost:80
echo API:      http://localhost:80/api/status
echo.
echo First startup takes 2-3 minutes to load the search model.
goto :eof

:up
echo Starting services...
docker compose up -d
echo.
echo Frontend: http://localhost:80
goto :eof

:update
echo Pulling latest images...
docker compose pull
echo Restarting services...
docker compose up -d
echo.
echo Frontend: http://localhost:80
goto :eof

:down
echo Stopping services...
docker compose down
goto :eof

:logs
shift
docker compose logs -f %1 %2 %3
goto :eof

:status
echo === Docker Services ===
docker compose ps
echo.
echo === API Health ===
curl -s http://localhost:80/api/status 2>nul || echo   Not reachable
goto :eof

:help
echo PDfiles — document search engine
echo.
echo Usage: pdfiles.bat ^<command^> [options]
echo.
echo Commands:
echo   deploy DATA_PATH    First-time setup (creates .env, builds, starts)
echo   up                  Start services
echo   update              Pull latest images and restart
echo   down                Stop services
echo   logs [SERVICE]      View logs
echo   status              Health check
echo   help                Show this help
goto :eof
