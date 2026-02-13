# PDfiles

![pdf](https://github.com/user-attachments/assets/0f42f82a-250a-413b-9410-cb6561efcb0d)

do you have lots of pdfs that you want to search through visually? do you have an nvidia gpu?  if yes, this is for you!

### Features

- search through pdfs by text description (not OCR)
  

https://github.com/user-attachments/assets/880c4caf-1367-4836-bc36-ef35524e1b2d



- search for all photos similar to a particular photo
  

https://github.com/user-attachments/assets/5ab6b907-5b08-4ef9-8ead-4b1b25510ce0


- can use OPT files to speed up indexing
- can export your index files to backup or share with others

## Quick Start

1. Install [Docker](https://docs.docker.com/get-docker/)
2. Configure:
   ```
   cp .env.example .env
   # Edit .env — set DATA_PATH to your documents folder
   ```
3. Start:
   ```bash
   # Pull pre-built images (fastest)
   docker compose pull && docker compose up -d

   # Or build from source
   ./pdfiles.sh deploy
   ```
   On Windows: `pdfiles.bat deploy`

4. Open http://localhost

First startup downloads the search model (~4 GB) and takes 2-3 minutes.

## Usage

| Command | Description |
|---------|-------------|
| `./pdfiles.sh up` | Start services |
| `./pdfiles.sh down` | Stop services |
| `./pdfiles.sh logs` | View logs |
| `./pdfiles.sh status` | Health check |
| `./pdfiles.sh backup` | Backup data |
| `./pdfiles.sh --help` | All commands |

## Requirements

- Docker and Docker Compose
- NVIDIA GPU (12+ GB VRAM) for indexing, or a prebuilt search index for CPU-only mode:
  ```bash
  docker compose -f docker-compose.cpu.yml up -d
  ```


## How it works

pdfiles scans through all of the pdfs that you mount in `DATA_PATH` and saves each page's essence in the form of vectors.  when you type something in search, these words get turned into vectors; and then the two sets of vectors get compared.  the resulting files are an ordered list of images that are closest to what you search.  this is what enables "find similar photos" as well.


## Architecture Overview

```
PDF Pages (517K)
  |
  v
[Bouncer] ---> VISUAL / TEXT_ONLY / UNCERTAIN
  |                (Tier 1: PyMuPDF text_ratio)
  |                (Tier 2: Surya layout detection)
  v
[Indexer] ---> ColQwen2.5 embed ---> 2D block pool ---> Qdrant (262x128d multi-vectors)
  |
  v
[Librarian] ---> mean-pool to 1x128d ---> K-Means ---> auto-label clusters
  |
  v
[Search UI] ---> text query ---> ColQwen2.5 ---> MaxSim ---> ranked results
                 cluster browse ---> gallery of representative pages
```
