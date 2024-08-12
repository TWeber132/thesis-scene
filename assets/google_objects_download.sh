#!/usr/bin/env bash
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

wget https://github.com/cliport/cliport/releases/download/v1.0.0/google.zip
unzip google.zip -d $SCRIPTPATH
rm google.zip