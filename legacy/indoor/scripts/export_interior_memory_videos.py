"""将服务器对比视频转成可直接预览的H.264，并核对帧数。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,subprocess,argparse
import cv2,imageio_ffmpeg

p=argparse.ArgumentParser();p.add_argument('--edge-retention-frames',type=int,default=30);p.add_argument('--interior-retention-frames',type=int);p.add_argument('--max-gap-frames',type=int,default=150);p.add_argument('--folder');p.add_argument('--videos',nargs='+',default=['B-5-1','B-5-3']);a=p.parse_args()
mode=f'offline_birth4_edge{a.edge_retention_frames}'
if a.interior_retention_frames is not None:mode+=f'_inside{a.interior_retention_frames}'
if a.max_gap_frames!=150:mode+=f'_fill{a.max_gap_frames}'
folder=Path(__file__).resolve().parents[1]/'artifacts/indoor_interior_memory_20260906'/mode
if a.folder:folder=Path(a.folder).resolve()
os.chdir(folder)
def convert(video):
    target=f'{video}_global_compare_10s.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-i',f'{video}_global_compare_source.mp4','-c:v','libx264','-preset','fast','-crf','18','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart','-an',target],check=True)
    cap=cv2.VideoCapture(target)
    result=dict(frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),fps=cap.get(cv2.CAP_PROP_FPS),width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    assert result==dict(frames=300,fps=30.,width=3840,height=1140)
    for frame in [0,150,299]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();assert ok
    cap.release();print(video,json.dumps(result),flush=True)
    return video,result
with ThreadPoolExecutor(2) as pool:results=dict(pool.map(convert,a.videos))
verification=Path('video_verification.json')
previous=json.loads(verification.read_text()) if verification.exists() else {}
previous.update(results)
verification.write_text(json.dumps(previous,indent=2))
