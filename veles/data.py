from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Condition:
    br_file: int
    values: dict[str, str]


@dataclass
class Session:
    name: str
    location: str
    conditions: list[Condition] = field(default_factory=list)
    status: dict[str, str] = field(default_factory=dict)
    # Steps whose file input (e.g. manual_trial_remove.csv) changed for this session since they last ran.
    stale_file_inputs: set[str] = field(default_factory=set)
