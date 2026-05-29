#!/bin/bash

# TinyLlama Model Download Script
# This script downloads the tinyllama model using Ollama

echo "🤖 Downloading TinyLlama model..."
echo "This may take a few minutes depending on your internet speed."
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    # Try the Homebrew path for macOS
    if [ -x "/opt/homebrew/bin/ollama" ]; then
        OLLAMA_CMD="/opt/homebrew/bin/ollama"
    else
        echo "❌ Error: Ollama is not installed."
        echo "Please install Ollama first:"
        echo "  macOS: brew install ollama"
        echo "  Linux/Windows: Visit https://ollama.ai"
        exit 1
    fi
else
    OLLAMA_CMD="ollama"
fi

# Download tinyllama model
echo "Downloading from Ollama registry..."
$OLLAMA_CMD pull tinyllama

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Success! TinyLlama model has been downloaded."
    echo ""
    echo "You can now use TinyLlama in the app:"
    echo "  1. Go to Profile"
    echo "  2. Click on AI Model"
    echo "  3. Select TinyLlama"
    echo "  4. Start using @gro to chat with AI!"
else
    echo ""
    echo "❌ Failed to download TinyLlama model."
    echo "Please check your internet connection and try again."
    exit 1
fi
