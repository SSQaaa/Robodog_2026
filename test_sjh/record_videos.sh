#!/usr/bin/env bash
# 注意要安装ffmpeg (sudo apt install ffmpeg)，然后要将存储介质（u盘、TF卡）挂载到系统上，并且有足够的空间 每 N 秒自动新建一个文件写到 TF 卡
# 用法: ./record_video.sh /dev/video0 /mnt/uu/videos 200 300
# 参数: $1 设备 (默认 /dev/video0) $2 输出目录 (默认 /mnt/uu/videos) $3 帧率 (默认 200) $4 分段时长(秒)，每段一个文件 (默认 300)

DEV=${1:-/dev/video0}
OUTDIR=${2:-/mnt/uu/videos}
FPS=${3:-25}
SEGSEC=${4:-300}

mkdir -p "$OUTDIR"

if [ "$SEGSEC" -le 0 ] 2>/dev/null; then
  echo "SEGSEC must be > 0" >&2
  exit 1
fi

echo "Recording... DEV=$DEV OUTDIR=$OUTDIR FPS=$FPS SEGSEC=$SEGSEC (Ctrl+C stop)"
i=0
while true; do
  TS=$(date +"%Y%m%d_%H%M%S")
  OUT="$OUTDIR/cam_${TS}_$(printf '%03d' "$i").mp4"
  ffmpeg -y -hide_banner -loglevel info \
    -f v4l2 -framerate "$FPS" -video_size 640x480 -i "$DEV" \
    -t "$SEGSEC" \
    -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p "$OUT"
  echo "Saved to: $OUT"
  i=$((i+1))
  sleep 1  # 避免频繁重启ffmpeg
done
