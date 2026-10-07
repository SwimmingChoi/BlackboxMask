"""Order-independent comparison of CPU and accelerator detections."""
def compare_boxes(reference, candidate):
    def iou(a,b):
        area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        return area/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)
    pairs=sorted(((iou(a['box'],b['box']),i,j) for i,a in enumerate(reference)
                  for j,b in enumerate(candidate) if a['kind']==b['kind']),reverse=True)
    used_a,used_b=set(),set()
    for value,i,j in pairs:
        if value>=.5 and i not in used_a and j not in used_b:
            used_a.add(i);used_b.add(j)
    total=max(len(reference),len(candidate))
    return {'reference_count':len(reference),'candidate_count':len(candidate),
            'agreement':len(used_a)/total if total else 1.,
            'missing_reference':[i for i in range(len(reference)) if i not in used_a],
            'extra_candidate':[j for j in range(len(candidate)) if j not in used_b]}
