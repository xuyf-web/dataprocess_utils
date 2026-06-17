#!/bin/bash

# init_project.sh - Create new project folder with structure matching current project

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <project_name>"
    exit 1
fi

PROJECT_NAME="$1"

if [ -d "$PROJECT_NAME" ]; then
    echo "Error: Directory '$PROJECT_NAME' already exists"
    exit 1
fi

echo "Creating project structure for: $PROJECT_NAME"

# Create main project directory
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Create top-level directories
mkdir -p archive
mkdir -p configs
mkdir -p data/processed
mkdir -p data/raw
mkdir -p docs/manuscripts
mkdir -p docs/publications
mkdir -p logs
mkdir -p model
mkdir -p reports
mkdir -p utils

# Create experiments directory
mkdir -p experiments

# Create basic files
cat > LICENSE << 'EOF'
MIT License

Copyright (c) $(date +%Y) [Yifei Xu, Nanjing University]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

touch README.md

# Create init_exp.sh in experiments directory
cat > experiments/init_exp.sh << 'EOF'
#!/bin/bash

# init_exp.sh - Create new experiment folder structure in experiments directory

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <experiment_number> <experiment_name>"
    echo "Example: $0 1 alpha"
    exit 1
fi

EXP_NUM="$1"
EXP_NAME="$2"
DATE=$(date +"%Y%m%d")
EXP_DIR="${EXP_NUM}.E${DATE}_${EXP_NAME}"

if [ -d "$EXP_DIR" ]; then
    echo "Error: Experiment directory '$EXP_DIR' already exists"
    exit 1
fi

echo "Creating experiment structure for: $EXP_DIR"

# Create experiment directory structure
mkdir -p "$EXP_DIR/cache"
mkdir -p "$EXP_DIR/figures"
mkdir -p "$EXP_DIR/logs"
mkdir -p "$EXP_DIR/src"
mkdir -p "$EXP_DIR/tests"
mkdir -p "$EXP_DIR/tmp"

# Create experiment README file
echo "# Experiment $EXP_NUM - ${EXP_NAME^}" > "$EXP_DIR/README.md"

echo "Experiment structure created successfully in: $EXP_DIR"
EOF

chmod +x experiments/init_exp.sh

echo "Project structure created successfully in: $PROJECT_NAME"
