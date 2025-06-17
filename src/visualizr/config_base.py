import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseConfig:
    """BaseConfig provides methods to clone itself,
    inherit settings from another config, propagate settings to nested configs,
    and serialize/deserialize configurations to/from JSON.
    """

    def clone(self):
        """Return a deep copy of this configuration."""
        return deepcopy(self)

    def inherit(self, another):
        """inherit common keys from a given config"""
        common_keys = set(self.__dict__.keys()) & set(another.__dict__.keys())
        for k in common_keys:
            setattr(self, k, getattr(another, k))

    def propagate(self):
        """push down the configuration to all members"""
        for _, v in self.__dict__.items():
            if isinstance(v, BaseConfig):
                v.inherit(self)
                v.propagate()

    def save(self, save_path: Path):
        """save config to JSON file"""
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        conf = self.as_dict_jsonable()
        with open(save_path, "w") as f:
            json.dump(conf, f)

    def load(self, load_path: Path):
        """load json config"""
        if not load_path.exists():
            load_path.mkdir(parents=True, exist_ok=True)
        with open(load_path) as f:
            conf = json.load(f)
        self.from_dict(conf)

    def from_dict(self, config_dict, strict=False):
        """Populate configuration attributes from a dictionary, optionally
        enforcing strict key checking.
        """
        for k, v in config_dict.items():
            if not hasattr(self, k):
                if strict:
                    raise ValueError(f"loading extra '{k}'")
                print(f"loading extra '{k}'")
                continue
            if isinstance(self.__dict__[k], BaseConfig):
                self.__dict__[k].from_dict(v)
            else:
                self.__dict__[k] = v

    def as_dict_jsonable(self):
        """Convert the configuration to a JSON-serializable dictionary."""
        conf = {}
        for k, v in self.__dict__.items():
            if isinstance(v, BaseConfig):
                conf[k] = v.as_dict_jsonable()
            else:
                if jsonable(v):
                    conf[k] = v
                else:
                    # ignore not jsonable
                    pass
        return conf


def jsonable(x):
    """Check if the object x is JSON serializable."""
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False
