def optimized_source(source):
 old="""            for row,j in zip(ix,detix):
                a2,l2,q2=pose(now,preds[j,:2])
                reliability=np.clip((min(quality,q2)-.15)/.45,0,1)*np.exp(-max(0,gap-1)/15)
                # Soft, bounded penalty: a turn alone never rejects a detection.
                angular=.5*(1-np.cos(wrap(a2-expected)))/2
                lengthpen=.12*min(abs(np.log(max(l2,1)/max(length,1))),1)
                d[row,4]+=reliability*(angular+lengthpen)"""
 new="""            a2,l2,q2=current_poses[detix].T
            reliability=np.clip((np.minimum(quality,q2)-.15)/.45,0,1)*np.exp(-max(0,gap-1)/15)
            angular=.5*(1-np.cos(wrap(a2-expected)))/2
            lengthpen=.12*np.minimum(np.abs(np.log(np.maximum(l2,1)/max(length,1))),1)
            d[ix,4]+=reliability*(angular+lengthpen)"""
 assert source.count(old)==1
 source=source.replace(old,new)
 source=source.replace('    now=int(fr_i[0])\n','    now=int(fr_i[0])\n    current_poses=np.array([pose(now,p[:2]) for p in preds])\n',1)
 old="""        for row in ix:
            j=int(d[row,1]);lastxy=xy[-1];current=preds[j,:2].astype(float)
            previous_point=lastxy;current_point=current
            residual=float(np.linalg.norm(current_point-(previous_point+shift)))
            if residual>radius:
                keep[row]=False"""
 new="""        js=d[ix,1].astype(int)
        residual=np.linalg.norm(preds[js,:2]-(xy[-1]+shift),axis=1)
        keep[ix]=residual<=radius"""
 assert source.count(old)==1
 return source.replace(old,new)
