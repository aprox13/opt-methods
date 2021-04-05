import json
from typing import Optional

import numpy as np

from simplex.solver import SimplexCanonicalIssue


def parse_int_safe(x):
    try:
        return int(x)
    except ValueError:
        return None


def parse_ints(items):
    return list(map(parse_int_safe, items))


def parse_enum(key, data, allowed_values, name_opt):
    if key in data:
        assert data[key] in allowed_values

        return [data[key]]
    else:
        if name_opt is not None:
            raise RuntimeError(f'You should specify {key} for {name_opt}')
        return allowed_values


class IssueRegistry:

    @staticmethod
    def load(path):

        res = IssueRegistry()

        js = json.load(open(path, 'r'))
        assert 'samples' in js

        for sample in js['samples']:

            assert 'A' in sample
            assert 'f' in sample
            assert 'b' in sample

            name_opt = None
            if 'name' in sample:
                name_opt = sample['name']

            A = np.array(sample['A'])
            c = np.array(sample['f'])
            b = np.array(sample['b'])

            types = parse_enum('type', sample, {'eq', 'lte'}, name_opt)
            targets = parse_enum('target', sample, {'min', 'max'}, name_opt)

            for t in types:
                for targ in targets:
                    if t == 'eq':
                        i = SimplexCanonicalIssue(A, b, c)
                    else:
                        i = SimplexCanonicalIssue.from_lte(A, b, c)
                    res.add_issue(i, targ, name_opt)

        return res

    def __init__(self):
        self._registry = []
        self._name_to_index = dict()

    def add_issue(self, issue: SimplexCanonicalIssue, target: str, name: Optional[str] = None):
        self._registry.append((issue, target))
        if name is not None:
            if name in self._name_to_index:
                raise KeyError(f"Duplicate name {name}")
            self._name_to_index[name] = len(self._registry) - 1

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= len(self._registry):
                raise KeyError(f'Index out of bound. There is only {len(self._registry)} issues stored')

            return self._registry[item]

        if isinstance(item, str):
            if item in self._name_to_index:
                return self._registry[self._name_to_index[item]]

            raise KeyError(f'Issue with name {item} not stored')

        raise KeyError(f'Unsupported key {item} of type {type(item)}. Supported only int and string')
