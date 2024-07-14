#!/bin/bash

style_file="/data/jittor2024/Problem2/JDiffusion/examples/dreambooth/settings/style.json"
texture_file="/data/jittor2024/Problem2/JDiffusion/examples/dreambooth/settings/texture.json"
keys=$(jq -r 'keys[]' "$style_file")
MAX_NUM=5
GPU_COUNT=1
#for key in $keys; do
#  value=$(jq -r --arg key "$key" '.[$key]' "$json_file")
#  echo "Key: $key, Value: $value"
#done

for ((folder_number = 0; folder_number <= $MAX_NUM; folder_number+=$GPU_COUNT)); do
  key=$(printf "%02d" $folder_number)
  style_prompt=$(jq -r --arg key "$key" '.[$key]' "$style_file")
  texture_prompt=$(jq -r --arg key "$key" '.[$key]' "$texture_file")
#  echo "Key: $key, style_prompt: $style_prompt, texture_prompt: $texture_prompt"
  echo "in $style_prompt style, with a texture of $texture_prompt."
done