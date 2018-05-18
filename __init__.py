import re

fregex = re.compile(r'\[([^\]]+)\]')
features = fregex.findall('[f1] = [f1] + [f2]')
print(features)