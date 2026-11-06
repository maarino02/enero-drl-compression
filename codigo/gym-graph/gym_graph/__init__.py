# Deriva de gym_graph/__init__.py de ENERO (https://github.com/BNN-UPC/ENERO), licencia Apache 2.0.
# Modificado en 2026. Consulta los ficheros LICENSE y NOTICE.
from gymnasium.envs.registration import register


register(
    id='GraphEnv-v15',
    entry_point='gym_graph.envs:Env15',
)

register(
    id='GraphEnv-v16',
    entry_point='gym_graph.envs:Env16',
)

register(
    id='GraphEnv-v20',
    entry_point='gym_graph.envs:Env20',
)

register(
    id='GraphEnv-v17',
    entry_point='gym_graph.envs:Env17',
)
