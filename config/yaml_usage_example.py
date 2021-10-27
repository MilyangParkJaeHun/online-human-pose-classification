import yaml

with open('pose.yml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    for d in data:
        print(d['name'])
