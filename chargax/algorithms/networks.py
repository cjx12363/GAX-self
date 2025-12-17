import jax
import equinox as eqx
from typing import List
import distrax
import jax.numpy as jnp
import chex
        

class ActorNetwork(eqx.Module):
    """演员网络"""

    layers: list

    def __init__(self, key, in_shape, hidden_features: List[int], num_actions):
        keys = jax.random.split(key, len(hidden_features))
        self.layers = [
            eqx.nn.Linear(in_shape, hidden_features[0], key=keys[0])
        ]
        for i, feature in enumerate(hidden_features[:-1]):
            self.layers.append(eqx.nn.Linear(feature, hidden_features[i+1], key=keys[i]))
        self.layers.append(eqx.nn.Linear(hidden_features[-1], num_actions, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return distrax.Categorical(logits=self.layers[-1](x))
    
class CriticNetwork(eqx.Module):
    """
        具有单一输出的评论家网络
        例如用于在给定状态时输出V值
        或在给定状态和动作时输出Q值
    """
    layers: list

    def __init__(self, key, in_shape, hidden_layers: List[int]):
        keys = jax.random.split(key, len(hidden_layers))
        self.layers = [ # 用第一层初始化
            eqx.nn.Linear(in_shape, hidden_layers[0], key=keys[0])
        ]
        for i, feature in enumerate(hidden_layers[:-1]):
            self.layers.append(eqx.nn.Linear(feature, hidden_layers[i+1], key=keys[i]))
        self.layers.append(eqx.nn.Linear(hidden_layers[-1], 1, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return jnp.squeeze(self.layers[-1](x), axis=-1)

class Q_CriticNetwork(eqx.Module):
    """
        为每个动作输出值的评论家网络
        例如Q值列表
    """

    layers: list

    def __init__(self, key, in_shape, hidden_layers: List[int], num_actions):
        keys = jax.random.split(key, len(hidden_layers))
        self.layers = [
            eqx.nn.Linear(in_shape, hidden_layers[0], key=keys[0])
        ]
        for i, feature in enumerate(hidden_layers[:-1]):
            self.layers.append(eqx.nn.Linear(feature, hidden_layers[i+1], key=keys[i]))
        self.layers.append(eqx.nn.Linear(hidden_layers[-1], num_actions, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x)
    
class ActorNetworkMultiDiscrete(eqx.Module):
    """
    用于多离散输出空间的演员网络
    """

    layers: list
    output_heads: list

    def __init__(self, key, in_shape, hidden_layers, actions_nvec):

        if isinstance(actions_nvec, chex.Array):
            actions_nvec = actions_nvec.tolist()

        keys = jax.random.split(key, len(hidden_layers))
        self.layers = [eqx.nn.Linear(in_shape, hidden_layers[0], key=keys[0])]
        for i, feature in enumerate(hidden_layers[:-1]):
            self.layers.append(
                eqx.nn.Linear(feature, hidden_layers[i + 1], key=keys[i])
            )

        multi_discrete_heads_keys = jax.random.split(keys[-1], len(actions_nvec))
        self.output_heads = [
            eqx.nn.Linear(hidden_layers[-1], action, key=multi_discrete_heads_keys[i])
            for i, action in enumerate(actions_nvec)
        ]
        if len(set(actions_nvec)) == 1:  # 所有输出形状相同，使用vmap
            self.output_heads = jax.tree_util.tree_map(
                lambda *v: jnp.stack(v), *self.output_heads
            )
        else:
            raise NotImplementedError(
                "检测到不同的输出形状。调用函数尚未考虑这种情况"
            )

    def __call__(self, x):

        def forward(head, x):
            return head(x)

        for layer in self.layers:
            x = jax.nn.tanh(layer(x))
        logits = jax.vmap(forward, in_axes=(0, None))(self.output_heads, x)

        return distrax.Categorical(logits=logits)
    
def create_actor_critic_network(
        key,
        in_shape: int,
        actor_features: List[int],
        critic_features: List[int],
        num_env_actions: int,
    ):
    """
        创建演员和评论家网络
    """
    actor_key, critic_key = jax.random.split(key, 2)
    actor = ActorNetwork(actor_key, in_shape, actor_features, num_env_actions)
    critic = CriticNetwork(critic_key, in_shape, critic_features)
    return actor, critic

def create_actor_critic_critic_network(
        key,
        in_shape: int,
        actor_features: List[int],
        critic_features: List[int],
        num_env_actions: int,
    ):
    """
        创建演员、2个评论家和目标网络（例如用于SAC）
    """
    actor_key, critic1_key, critic2_key = jax.random.split(key, 3)
    actor = ActorNetwork(actor_key, in_shape, actor_features, num_env_actions)
    critic = Q_CriticNetwork(critic1_key, in_shape, critic_features, num_env_actions)
    critic2 = Q_CriticNetwork(critic2_key, in_shape, critic_features, num_env_actions)
    critic1_target = jax.tree.map(lambda x: x, critic)
    critic2_target = jax.tree.map(lambda x: x, critic2)
    return actor, critic, critic2, critic1_target, critic2_target