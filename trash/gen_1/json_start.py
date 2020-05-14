import json

with open('setopt/2020-04-16.json', 'r') as f:
    distros_dict = json.load(f)

for distro in distros_dict:
    print(distro['Name'])



