"""Tests for JAX-based game implementation."""
import jax
import jax.numpy as jnp
import pytest

from generals import GeneralsEnv
from generals.core import game


def create_test_grid(size=4):
    """Create a simple test grid with generals in corners."""
    grid = jnp.zeros((size, size), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)  # General player 0
    grid = grid.at[size - 1, size - 1].set(2)  # General player 1
    return grid


def test_create_initial_state():
    """Test creating initial state from a grid."""
    grid = create_test_grid(4)
    state = game.create_initial_state(grid)

    # Check state structure
    assert hasattr(state, "armies")
    assert hasattr(state, "ownership")
    assert hasattr(state, "generals")
    assert hasattr(state, "time")
    assert hasattr(state, "winner")

    # Check initial armies
    assert state.armies[0, 0] == 1  # General A
    assert state.armies[3, 3] == 1  # General B

    # Check ownership
    assert state.ownership[0, 0, 0] == True  # Player 0 owns (0,0)
    assert state.ownership[1, 3, 3] == True  # Player 1 owns (3,3)

    # Check initial game state
    assert state.time == 0
    assert state.winner == -1


def test_generals_io_profile_is_explicit_public_variable_canvas():
    env = GeneralsEnv(mode="generals-io", pool_size=4)

    assert env.mode == "generals-io"
    assert (env.min_grid_size, env.max_grid_size, env.pad_to) == (17, 23, 24)
    assert env.truncation == 2048
    assert env.perfect_info is False


def test_generals_io_observation_exposes_public_unpadded_extent():
    env = GeneralsEnv(mode="generals-io", pool_size=49)
    pool, _ = env.reset(jax.random.PRNGKey(23))
    observation = jax.vmap(lambda state: game.get_observation(state, 0))(pool)

    assert observation.extent is not None
    assert observation.extent.shape == (49, 24, 24)
    expected = sorted(h * w for h in range(17, 24) for w in range(17, 24))
    assert sorted(jnp.sum(observation.extent, axis=(1, 2)).tolist()) == expected


def test_step_pass_action():
    """Test that pass actions don't change state."""
    grid = create_test_grid(2)
    state = game.create_initial_state(grid)

    # Both players pass
    actions = jnp.array(
        [
            [1, 0, 0, 0, 0],  # Player 0 passes
            [1, 0, 0, 0, 0],  # Player 1 passes
        ],
        dtype=jnp.int32,
    )

    new_state, info = game.step(state, actions)

    # Armies should not change (except time increment)
    assert jnp.array_equal(new_state.armies, state.armies)
    assert new_state.time == 1
    assert new_state.winner == -1


def test_step_move_to_neutral():
    """Test moving to a neutral cell."""
    grid = create_test_grid(3)
    state = game.create_initial_state(grid)

    # Give player 0 more armies
    state = state._replace(armies=state.armies.at[0, 0].set(5))

    # Player 0 moves right (direction 3)
    actions = jnp.array(
        [
            [0, 0, 0, 3, 0],  # Move right from (0,0)
            [1, 0, 0, 0, 0],  # Player 1 passes
        ],
        dtype=jnp.int32,
    )

    new_state, info = game.step(state, actions)

    # Check armies moved
    assert new_state.armies[0, 0] == 1  # Left 1 behind
    assert new_state.armies[0, 1] == 4  # Moved 4

    # Check ownership changed
    assert new_state.ownership[0, 0, 1] == True


def test_step_move_to_own_cell():
    """Test moving to own cell (merge armies)."""
    grid = create_test_grid(3)
    state = game.create_initial_state(grid)

    # Setup: Give player 0 two cells with armies
    state = state._replace(
        armies=state.armies.at[0, 0].set(5).at[0, 1].set(3),
        ownership=state.ownership.at[0, 0, 1].set(True),
        ownership_neutral=state.ownership_neutral.at[0, 1].set(False),
    )

    # Player 0 moves from (0,0) to (0,1)
    actions = jnp.array(
        [
            [0, 0, 0, 3, 0],  # Move right
            [1, 0, 0, 0, 0],  # Pass
        ],
        dtype=jnp.int32,
    )

    new_state, info = game.step(state, actions)

    # Armies should merge
    assert new_state.armies[0, 0] == 1  # Left 1 behind
    assert new_state.armies[0, 1] == 7  # 3 + 4 moved


def test_get_observation():
    """Test observation generation with fog of war."""
    grid = create_test_grid(4)
    state = game.create_initial_state(grid)

    obs = game.get_observation(state, 0)

    # Check observation structure
    assert hasattr(obs, "armies")
    assert hasattr(obs, "owned_cells")
    assert hasattr(obs, "fog_cells")
    assert hasattr(obs, "timestep")

    # Player 0 should see their general
    assert obs.armies[0, 0] == 1

    # Player 0 should not see player 1's general (too far)
    assert obs.armies[3, 3] == 0  # Hidden in fog


def test_global_update():
    """Test global army increment mechanics."""
    grid = create_test_grid(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(5), time=jnp.int32(2))
    state = game.global_update(state)

    # General should have gained 1 army
    assert state.armies[0, 0] == 6


def test_captured_general_converts_to_city_and_halves_defeated_armies():
    """Terminal capture follows the public generals.io transfer contract."""
    grid = jnp.zeros((3, 3), dtype=jnp.int32)
    grid = grid.at[1, 0].set(1)
    grid = grid.at[1, 1].set(2)
    state = game.create_initial_state(grid)._replace(
        armies=jnp.array(
            [[0, 0, 0], [9, 3, 0], [0, 7, 0]], dtype=jnp.int32
        ),
        ownership=jnp.array(
            [
                [[False, False, False], [True, False, False], [False, False, False]],
                [[False, False, False], [False, True, False], [False, True, False]],
            ]
        ),
        ownership_neutral=jnp.array(
            [[False, True, True], [True, False, True], [True, False, True]]
        ),
    )
    actions = jnp.array(
        [[0, 1, 0, 3, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32
    )

    new_state, info = game.step(state, actions)

    assert int(info.winner) == 0
    assert bool(new_state.cities[1, 1])
    assert not bool(new_state.generals[1, 1])
    assert bool(new_state.ownership[0, 1, 1])
    assert not bool(new_state.ownership[1, 1, 1])
    # The defeated owned stack of 7 becomes ceil-half 4.  The attacking
    # winner stack is not halved.
    assert int(new_state.armies[2, 1]) == 4
    assert int(new_state.armies[1, 1]) == 5


def test_batch_step():
    """Test batched step execution."""
    grid = create_test_grid(2)
    state = game.create_initial_state(grid)

    # Stack into batch
    batched_state = jax.tree.map(lambda x: jnp.stack([x, x]), state)

    # Create actions for both envs
    actions = jnp.array(
        [
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],  # Env 0: both pass
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]],  # Env 1: both pass
        ],
        dtype=jnp.int32,
    )

    new_states, infos = game.batch_step(batched_state, actions)

    # Check batch dimension preserved
    assert new_states.time.shape == (2,)
    assert new_states.armies.shape == (2, 2, 2)


def test_jit_compilation():
    """Test that step function can be JIT compiled."""
    grid = create_test_grid(2)
    state = game.create_initial_state(grid)

    # JIT compile step
    jitted_step = jax.jit(game.step)

    actions = jnp.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )

    # Should execute without errors
    new_state, info = jitted_step(state, actions)

    assert new_state.time == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
