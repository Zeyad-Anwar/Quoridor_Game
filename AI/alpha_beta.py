"""Alpha-beta pruning agent for Quoridor.

This is intended as a lightweight baseline / bootstrap policy for early training.
It does NOT use the neural network.

Design goals:
- Works with the existing `game.GameState` API (clone + apply_action)
- Keeps branching factor reasonable by considering a limited set of wall candidates
- Returns an action distribution compatible with the AlphaZero training pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import math
import numpy as np

from constants import ACTION_SPACE_SIZE
from game import GameState, Player, Position, Wall
from AI.action_utils import action_to_index, index_to_action, action_to_string


@dataclass(frozen=True)
class AlphaBetaConfig:
    depth: int = 2
    max_moves_per_position: int = 32
    max_wall_candidates: int = 24
    wall_neighborhood: int = 2
    # Heuristic weights
    w_dist: float = 1.0
    w_walls: float = 0.15
    # Convert heuristic score to [-1, 1]
    value_scale: float = 0.35


def _shortest_path_len_ignoring_opponent(state: GameState, player: Player) -> int:
    """Shortest path length in the wall graph, ignoring the opponent pawn.

    Uses simple 4-neighbor movement with wall blocking (no jump/diagonal rules).
    This is a common heuristic approximation for Quoridor.
    """
    start = state.get_player_pos(player)
    goal_row = 0 if player == 1 else 8

    if start[0] == goal_row:
        return 0

    # BFS on 9x9 grid
    dist = [-1] * (9 * 9)
    start_idx = start[0] * 9 + start[1]
    dist[start_idx] = 0

    queue: list[Position] = [start]
    qi = 0
    walls = state.walls

    while qi < len(queue):
        r, c = queue[qi]
        qi += 1
        base_d = dist[r * 9 + c]

        # Up
        if r > 0:
            nr = r - 1
            idx = nr * 9 + c
            if dist[idx] == -1:
                wall_row = nr
                blocked = False
                if c > 0 and ((wall_row, c - 1), "H") in walls:
                    blocked = True
                elif c < 8 and ((wall_row, c), "H") in walls:
                    blocked = True
                if not blocked:
                    if nr == goal_row:
                        return base_d + 1
                    dist[idx] = base_d + 1
                    queue.append((nr, c))

        # Down
        if r < 8:
            nr = r + 1
            idx = nr * 9 + c
            if dist[idx] == -1:
                wall_row = r
                blocked = False
                if c > 0 and ((wall_row, c - 1), "H") in walls:
                    blocked = True
                elif c < 8 and ((wall_row, c), "H") in walls:
                    blocked = True
                if not blocked:
                    if nr == goal_row:
                        return base_d + 1
                    dist[idx] = base_d + 1
                    queue.append((nr, c))

        # Left
        if c > 0:
            nc = c - 1
            idx = r * 9 + nc
            if dist[idx] == -1:
                wall_col = nc
                blocked = False
                if r > 0 and ((r - 1, wall_col), "V") in walls:
                    blocked = True
                elif r < 8 and ((r, wall_col), "V") in walls:
                    blocked = True
                if not blocked:
                    dist[idx] = base_d + 1
                    queue.append((r, nc))

        # Right
        if c < 8:
            nc = c + 1
            idx = r * 9 + nc
            if dist[idx] == -1:
                wall_col = c
                blocked = False
                if r > 0 and ((r - 1, wall_col), "V") in walls:
                    blocked = True
                elif r < 8 and ((r, wall_col), "V") in walls:
                    blocked = True
                if not blocked:
                    dist[idx] = base_d + 1
                    queue.append((r, nc))

    # Unreachable shouldn't happen for legal positions, but keep stable.
    return 99


def _evaluate(state: GameState, cfg: AlphaBetaConfig, perspective: Player) -> float:
    """Heuristic value in [-1, 1] from `perspective` player's POV."""
    if state.winner is not None:
        if state.winner == perspective:
            return 1.0
        return -1.0

    my_dist = _shortest_path_len_ignoring_opponent(state, perspective)
    opp = state.get_opponent(perspective)
    opp_dist = _shortest_path_len_ignoring_opponent(state, opp)

    # Critical: heavily favor being very close to winning
    # If we're 1 move from winning, this should be near-winning value
    if my_dist == 1 and opp_dist > 1:
        return 0.95  # Almost winning!
    if opp_dist == 1 and my_dist > 1:
        return -0.95  # Opponent almost winning!

    my_walls = state.get_walls_remaining(perspective)
    opp_walls = state.get_walls_remaining(opp)

    # Larger is better
    raw = cfg.w_dist * (opp_dist - my_dist) + cfg.w_walls * (my_walls - opp_walls)

    # Squash to [-1, 1]
    return float(math.tanh(cfg.value_scale * raw))


def _candidate_walls(state: GameState, cfg: AlphaBetaConfig) -> list[Wall]:
    """Generate a small set of plausible wall candidates near pawns."""
    if state.get_walls_remaining(state.current_player) <= 0:
        return []

    candidates: list[Wall] = []

    def add_neighborhood(center: Position) -> None:
        r, c = center
        # Wall coords are 0..7, and are between tiles.
        r0 = max(0, r - cfg.wall_neighborhood)
        r1 = min(7, r + cfg.wall_neighborhood)
        c0 = max(0, c - cfg.wall_neighborhood)
        c1 = min(7, c + cfg.wall_neighborhood)
        for wr in range(r0, r1 + 1):
            for wc in range(c0, c1 + 1):
                for o in ("H", "V"):
                    w: Wall = ((wr, wc), o)
                    if state.is_valid_wall_placement(w):
                        candidates.append(w)

    me = state.get_player_pos(state.current_player)
    opp = state.get_player_pos(state.get_opponent(state.current_player))
    add_neighborhood(opp)
    add_neighborhood(me)

    # Deduplicate while preserving order
    seen: set[Wall] = set()
    uniq: list[Wall] = []
    for w in candidates:
        if w not in seen:
            seen.add(w)
            uniq.append(w)

    # Keep only the most impactful walls (by immediate increase in opponent path length)
    if len(uniq) <= cfg.max_wall_candidates:
        return uniq

    opp_player = state.get_opponent(state.current_player)
    base_opp_dist = _shortest_path_len_ignoring_opponent(state, opp_player)

    scored: list[tuple[float, Wall]] = []
    for w in uniq:
        state.walls.add(w)
        try:
            new_opp_dist = _shortest_path_len_ignoring_opponent(state, opp_player)
        finally:
            state.walls.remove(w)
        scored.append((new_opp_dist - base_opp_dist, w))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [w for _, w in scored[: cfg.max_wall_candidates]]


def _candidate_actions(state: GameState, cfg: AlphaBetaConfig) -> list[tuple[str, Position | Wall]]:
    actions: list[tuple[str, Position | Wall]] = []

    # Always include pawn moves.
    for pos in state.get_valid_moves(state.current_player):
        actions.append(("move", pos))

    # Include a limited set of wall candidates.
    for w in _candidate_walls(state, cfg):
        actions.append(("wall", w))

    # If something went wrong and we have no candidates, fall back to full legal actions.
    if not actions:
        return state.get_legal_actions()

    return actions


def _negamax(
    state: GameState,
    cfg: AlphaBetaConfig,
    depth: int,
    alpha: float,
    beta: float,
    tt: dict[int, tuple[int, float]],
) -> float:
    """Negamax with alpha-beta pruning. Returns value from current player's POV.
    
    Note: In terminal states where a player just won, current_player is still the winner
    (turn doesn't switch on game end). We need to return the value as if the opponent
    were to play (i.e., -1 for them), so the parent's negation gives +1 for the winner.
    """
    if state.is_terminal():
        # Winner just moved and current_player == winner (turn didn't switch)
        # Return from the hypothetical "next player" (opponent's) perspective
        # so parent's negation works correctly
        if state.winner == state.current_player:
            # Opponent would see this as a loss: -1
            # Parent will negate: -(-1) = +1 (correct: winning move is good)
            return -1.0
        else:
            # Current player lost (shouldn't normally happen in standard play)
            return 1.0

    current = state.current_player
    
    if depth <= 0:
        return _evaluate(state, cfg, current)

    h = hash(state)
    cached = tt.get(h)
    if cached is not None:
        cached_depth, cached_val = cached
        if cached_depth >= depth:
            return cached_val

    best = -1e9

    actions = _candidate_actions(state, cfg)

    # Simple move ordering: prefer actions that improve heuristic immediately
    scored_actions: list[tuple[float, tuple[str, Position | Wall]]] = []
    for a in actions:
        child = state.clone()
        child.apply_action(a)
        # Evaluate from opponent's perspective (child.current_player), then negate
        scored_actions.append((-_evaluate(child, cfg, child.current_player), a))

    scored_actions.sort(key=lambda t: t[0], reverse=True)

    # Limit branching
    for _, action in scored_actions[: cfg.max_moves_per_position]:
        child = state.clone()
        child.apply_action(action)

        val = -_negamax(child, cfg, depth - 1, -beta, -alpha, tt)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    tt[h] = (depth, best)
    return best


def alphabeta_action_probs(
    state: GameState,
    cfg: Optional[AlphaBetaConfig] = None,
    temperature: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Return (policy, value) for the current player using alpha-beta.

    policy is a distribution over the global 209 action space.
    value is in [-1, 1] from the current player's perspective.
    """
    if cfg is None:
        cfg = AlphaBetaConfig()

    # Root perspective is the player to move.
    perspective = state.current_player

    # Terminal shortcut
    if state.is_terminal():
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        return policy, _evaluate(state, cfg, perspective)

    actions = _candidate_actions(state, cfg)

    # Score root actions via one-ply + deeper search
    tt: dict[int, tuple[int, float]] = {}
    action_scores: list[tuple[float, int]] = []

    alpha = -1e9
    beta = 1e9

    for action in actions[: cfg.max_moves_per_position]:
        child = state.clone()
        child.apply_action(action)

        score = -_negamax(child, cfg, cfg.depth - 1, -beta, -alpha, tt)
        idx = action_to_index(action)
        action_scores.append((score, idx))

        if score > alpha:
            alpha = score

    # Convert scores to probabilities
    policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    if not action_scores:
        return policy, 0.0

    # Greedy
    if temperature <= 0:
        best_idx = max(action_scores, key=lambda t: t[0])[1]
        policy[best_idx] = 1.0
        return policy, float(max(action_scores, key=lambda t: t[0])[0])

    # Softmax sampling over evaluated moves
    raw_scores = np.array([s for s, _ in action_scores], dtype=np.float32)
    # Shift for numerical stability in softmax (doesn't affect probabilities)
    shifted_scores = raw_scores - float(raw_scores.max())
    probs = np.exp(shifted_scores / float(temperature))
    probs_sum = float(probs.sum())
    if probs_sum <= 0 or not math.isfinite(probs_sum):
        # Fallback uniform over considered actions
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / probs_sum

    for p, (_, idx) in zip(probs, action_scores):
        policy[idx] = float(p)

    # Use original (unshifted) scores for expected value calculation
    root_value = float(np.dot(raw_scores, probs))
    return policy, root_value


class AlphaBetaPlayer:
    """Gameplay wrapper for alpha-beta + heuristic (human vs AI)."""

    def __init__(
        self,
        player: int,
        cfg: Optional[AlphaBetaConfig] = None,
        temperature: float = 0.0,
    ):
        self.player = player
        self.cfg = cfg or AlphaBetaConfig()
        self.temperature = temperature

    def get_action(self, state: GameState) -> tuple:
        """Get an action for the given state."""
        if os.environ.get("QUORIDOR_DEBUG_LEGAL_MOVES", "0") == "1":
            current = state.current_player
            pawn_moves = state.get_valid_moves(current)
            goal_row = 0 if current == 1 else 8
            winning_moves = [m for m in pawn_moves if m[0] == goal_row]

            print(
                f"[AlphaBeta] Player {current} legal pawn moves ({len(pawn_moves)}): {pawn_moves}",
                flush=True,
            )
            if winning_moves:
                print(
                    f"[AlphaBeta] Player {current} has winning pawn move(s): {winning_moves}",
                    flush=True,
                )

            # Optional: also print full legal actions (moves + walls) for deeper debugging.
            if os.environ.get("QUORIDOR_DEBUG_LEGAL_ACTIONS", "0") == "1":
                legal_actions = state.get_legal_actions()
                pretty = ", ".join(action_to_string(a) for a in legal_actions)
                print(
                    f"[AlphaBeta] Player {current} legal actions ({len(legal_actions)}): {pretty}",
                    flush=True,
                )

        policy, _ = alphabeta_action_probs(
            state, cfg=self.cfg, temperature=self.temperature
        )
        if policy.sum() <= 0:
            # Fallback: pick the first legal action
            return state.get_legal_actions()[0]

        if self.temperature <= 0:
            action_idx = int(np.argmax(policy))
        else:
            action_idx = int(np.random.choice(len(policy), p=policy))

        return index_to_action(action_idx)
