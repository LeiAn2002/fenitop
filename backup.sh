#!/bin/bash
DATE=$(date +%Y-%m-%d)
git add .
git commit -m "daily_update $DATE"
git push -f origin cloak