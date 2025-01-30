from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class TimeConstraint:
    type: str
    confidence: float

@dataclass
class SubTask:
    text: str
    type: str
    confidence: float
    related_tasks: List[Tuple[str, float]] = field(default_factory=list)

@dataclass
class Swift799Message:
    original_text: str
    main_task: str
    has_conditional_actions: bool
    sub_tasks: List[SubTask] = field(default_factory=list)
    time_constraints: List[TimeConstraint] = field(default_factory=list)
