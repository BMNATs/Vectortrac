import pandas as pd
p = pd.read_html('stat_a66.xls', header=0, encoding='utf-8')[0]
c = p.columns
sub = p[p[c[2]].astype(str).str.contains('ปทุมธานี')]
print(sub[[c[2], c[3], c[4], c[11], c[12]]].to_string())