"""透明压缩JSON中间结果，不改变框坐标及处理规则。"""
import gzip,json,os
from pathlib import PosixPath,Path

class JsonPath(PosixPath):
    def packed(self):return Path(str(self)+'.gz')
    def exists(self):return super().exists() or self.suffix=='.json' and self.packed().exists()
    def read_text(self,*args,**kwargs):
        if self.suffix=='.json' and self.packed().exists():
            with gzip.open(self.packed(),'rt',encoding='utf-8') as f:return f.read()
        return super().read_text(*args,**kwargs)
    def write_text(self,data,*args,**kwargs):
        if self.suffix!='.json':return super().write_text(data,*args,**kwargs)
        dest=self.packed();tmp=Path(str(dest)+'.tmp')
        with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=1) as f:f.write(data)
        os.replace(tmp,dest);return len(data)
    def glob(self,pattern):
        found={str(p):p for p in super().glob(pattern)}
        if pattern.endswith('.json'):
            for p in super().glob(pattern+'.gz'):found[str(p)[:-3]]=JsonPath(str(p)[:-3])
        return iter(found.values())

def save(path,obj):
    path=JsonPath(path);path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(obj,dict) and 'detections' in obj:
        # 已删除框的大型历史列表留在原始结果中，当前阶段只保留处理计数和有效检测。
        obj={k:v for k,v in obj.items() if not k.endswith('_removed') and not k.endswith('_removed_detections') and k not in ['removed_detections','density_centers','density_counts','reassociation_candidates','reassociation_overlap_rejected']}
    path.write_text(json.dumps(obj,separators=(',',':')))
