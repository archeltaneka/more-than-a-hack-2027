"""Fixed physical/time priors; no parameters are estimated from public labels."""
from dataclasses import asdict, dataclass, replace
import math


@dataclass(frozen=True)
class Config:
    fps: int = 25
    bridge_seconds: float = 2.0
    bridge_weight: float = 0.25
    jump_limit_mps: float = 50.0
    top_k: int = 3
    proximity_m: float = 3.0
    nearest_weight: float = 0.5
    smoothing_seconds: float = 0.5
    motion_seconds: float = 0.5
    slow_mps: float = 1.0
    displacement_m: float = 1.0
    distant_player_m: float = 5.0
    speed_weight: float = 0.35
    displacement_weight: float = 0.25
    distance_weight: float = 0.15
    tag_weight: float = 0.25
    dead_reference: float = 0.55
    team_weight: float = 1.0
    dead_weight: float = 2.0
    team_change_penalty: float = 1.0
    dead_change_penalty: float = 0.75

    def __post_init__(self):
        values = asdict(self)
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError('configuration values must be finite and nonnegative')
        for name in ('fps', 'top_k'):
            if type(values[name]) is not int or values[name] < 1:
                raise ValueError(name+' must be a positive integer')
        for name in ('jump_limit_mps', 'proximity_m', 'slow_mps', 'displacement_m', 'distant_player_m'):
            if values[name] <= 0:
                raise ValueError(name+' must be positive')
        for name in ('bridge_weight', 'nearest_weight', 'dead_reference'):
            if values[name] > 1:
                raise ValueError(name+' must be in [0,1]')
        if not math.isclose(self.speed_weight+self.displacement_weight+self.distance_weight+self.tag_weight, 1):
            raise ValueError('dead cue weights must sum to one')


def variations(config):
    """Predeclared one-factor sensitivity; never score/select using public labels."""
    return {
        'penalties_half': replace(config, team_change_penalty=config.team_change_penalty*.5,
                                  dead_change_penalty=config.dead_change_penalty*.5),
        'penalties_double': replace(config, team_change_penalty=config.team_change_penalty*2,
                                    dead_change_penalty=config.dead_change_penalty*2),
        'bridge_1s': replace(config, bridge_seconds=1),
        'bridge_4s': replace(config, bridge_seconds=4),
        'proximity_2m': replace(config, proximity_m=2),
        'proximity_5m': replace(config, proximity_m=5),
        'dead_weight_half': replace(config, dead_weight=config.dead_weight*.5),
        'dead_weight_1_5x': replace(config, dead_weight=config.dead_weight*1.5),
    }
