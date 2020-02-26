import yaml
from pathlib import Path
import htrc_features
from htrc_features import caching, Volume
import os
from compare_tools.resolver import resolver_factory, combine_resolvers

# You can change this here to get a different resolver.
default = """
resolver: 
  -
    id_resolver: pairtree
    dir: /drobo/feature-counts
    format: json
    compression: bz2
"""

try:
    for path in ["~/.htrc-config.yaml", "local.yaml"]:
        if os.path.exists(path):
            config = yaml.safe_load(Path(path).expanduser().open())
            break
    if not config:
        raise FileNotFoundError
except FileNotFoundError:
    raise
    config = yaml.safe_load(default)

resolver = config['resolver']
my_resolver = combine_resolvers(resolver)

if __name__ == "__main__":
    print(Volume(id="mdp.39015012434786", id_resolver = my_resolver).tokenlist(pos=False, section="default"))
