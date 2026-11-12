from __future__ import annotations
import argparse, json, random, shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('data/processed')); p.add_argument('--seed',type=int,default=42); p.add_argument('--train',type=float,default=.7); p.add_argument('--val',type=float,default=.2); a=p.parse_args(); assert a.train+a.val<1
 imgs=[x for x in a.source.rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}]; random.seed(a.seed); random.shuffle(imgs); n=len(imgs); cuts=[int(n*a.train),int(n*(a.train+a.val))]; manifest=[]
 for name,group in zip(('train','val','test'),(imgs[:cuts[0]],imgs[cuts[0]:cuts[1]],imgs[cuts[1]:])):
  d=a.output/name; d.mkdir(parents=True,exist_ok=True)
  for src in group:
   dst=d/src.name; shutil.copy2(src,dst); lab=src.with_suffix('.txt')
   if lab.exists(): shutil.copy2(lab,d/lab.name)
   manifest.append({'image':str(dst),'split':name,'label':str(d/lab.name) if lab.exists() else None})
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)); (a.output/'dataset_version.json').write_text(json.dumps({'version':'generated','source':str(a.source),'seed':a.seed,'splits':{'train':a.train,'val':a.val,'test':1-a.train-a.val},'image_count':n},indent=2))
 print(f'Prepared {n} images; no class assumptions were made.')
if __name__=='__main__': main()
