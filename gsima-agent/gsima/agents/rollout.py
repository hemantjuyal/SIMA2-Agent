"""Rollout planner for deterministic simulators.

Provides a simple breadth-first rollout search and utility scoring
as described in the architecture review.
"""
from typing import List, Dict, Any, Tuple

from gsima.utils import config


class RolloutPlanner:
    def __init__(self, adapter, depth: int = 3, turn_penalty: float = 0.1, collision_penalty: float = 1.0, goal_bonus: float = 10.0):
        self.adapter = adapter
        self.depth = depth
        self.turn_penalty = turn_penalty
        self.collision_penalty = collision_penalty
        self.goal_bonus = goal_bonus

    def _score_trajectory(self, initial_distance: int, trajectory: List[Dict[str, Any]]) -> float:
        """Compute a simple utility score for a trajectory.

        score = distance_gain - turn_penalty*turns - collision_penalty*collisions + goal_bonus (if reached)
        """
        if not trajectory:
            return -9999.0

        final = trajectory[-1]
        final_distance = final.get('distance_to_goal', initial_distance)
        distance_gain = max(0, initial_distance - final_distance)

        # Heuristics: count turns and collisions
        turns = 0
        collisions = 0
        for step in trajectory:
            if step.get('obstacle_in_front', '').lower() == 'true':
                collisions += 1
            # rough proxy: turning changes orientation but we can't detect turn vs move here; skip accurate counting

        score = distance_gain - (self.turn_penalty * turns) - (self.collision_penalty * collisions)
        if final_distance == 0:
            score += self.goal_bonus
        return score

    def plan(self) -> Tuple[str, List[Tuple[List[str], List[Dict[str, Any]], float]]]:
        """Run rollouts from current adapter state and return best action plus ranked trajectories.

        Returns:
            best_action_name, list of (actions_sequence, trajectory_states, score) sorted desc
        """
        # Get current state using adapter capabilities instead of assuming a fixed schema.
        state = self.adapter.get_current_env_state()
        start_pos = state.get('agent_pos')
        start_dir = state.get('agent_dir')

        # Progress signal is optional for some environments.
        progress = self.adapter.get_progress()
        initial_distance = progress.get('distance_to_goal') if isinstance(progress, dict) else None

        can_use_distance = self.adapter.supports_goal_distance()
        if initial_distance is None and can_use_distance and start_pos is not None:
            try:
                initial_distance = self.adapter.get_distance_to_goal(start_pos)
            except Exception:
                initial_distance = None

        canonical_actions = [a.name for a in self.adapter.get_canonical_actions()]

        # For rollouts, exclude STOP unless the adapter explicitly allows it and the
        # environment indicates the agent is close enough.
        can_use_stop = (
            self.adapter.supports_stop()
            and initial_distance is not None
            and initial_distance <= config.STOP_DISTANCE_THRESHOLD
        )
        if can_use_stop:
            include_stop = True
        else:
            include_stop = False

        if not include_stop and 'STOP' in canonical_actions:
            canonical_actions = [a for a in canonical_actions if a != 'STOP']

        # Generate all action sequences up to depth.
        sequences = [[]]
        for _ in range(self.depth):
            new_seqs = []
            for seq in sequences:
                for act in canonical_actions:
                    new_seqs.append(seq + [act])
            sequences = new_seqs

        evaluated = []
        can_simulate = (
            self.adapter.supports_simulation()
            and start_pos is not None
            and start_dir is not None
            and initial_distance is not None
        )
        if can_simulate:
            for seq in sequences:
                traj = self.adapter.simulate_trajectory(start_pos, start_dir, seq)
                score = self._score_trajectory(initial_distance, traj)
                evaluated.append((seq, traj, score))
        else:
            # Fallback: no simulator support, so keep the search shallow and rely on generic action order.
            for seq in sequences:
                evaluated.append((seq, [], -9999.0))

        evaluated.sort(key=lambda x: x[2], reverse=True)

        best_seq = evaluated[0] if evaluated else (['MOVE_FORWARD'], [], -9999.0)
        best_action = best_seq[0][0] if best_seq[0] else 'MOVE_FORWARD'
        return best_action, evaluated
