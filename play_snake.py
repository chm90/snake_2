import argparse
from snake_gym import make_snake
import sys
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import numpy as np
from pprint import pprint
from time import sleep
from run_snake import MyPolicy
from random import randint
def play_snake(model_path,
               policy,
               nsteps=100,
               nminibatches=4,
               ent_coef=.01,
               vf_coef=0.5,
               max_grad_norm=0.5,
               seed=1):
    nenvs = 1
    from baselines.ppo2.ppo2 import Model
    from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
    import tensorflow as tf
    import multiprocessing
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=ncpu,
        inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True  #pylint: disable=E1101
    tf.Session(config=config).__enter__()
    set_global_seeds(seed)
    def make_env(rank):
        def env_fn():
            env = make_snake(shape=(5,5))
            env.seed(randint(0,10000000))
            #print("---------------------------------------------------------------")
            #print(obs.shape)
            #print("---------------------------------------------------------------")
            return env
        return env_fn
    env = DummyVecEnv([make_env(i) for i in range(nenvs)])
    env = VecFrameStack(env, 4)
    policy = {
        'cnn': CnnPolicy,
        'lstm': LstmPolicy,
        'lnlstm': LnLstmPolicy
    }[policy]
    policy = MyPolicy
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    model = Model(
        policy=policy,
        ob_space=env.observation_space,
        ac_space=env.action_space,
        nbatch_act=nenvs,
        nbatch_train=nbatch_train,
        nsteps=nsteps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm)
    model.load(model_path)
    player = Player(env=env,model=model)
    player.play()
    print("done")


class Player(object):

    def __init__(self, *, env, model,max_steps=1000):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.max_steps = max_steps

    def play(self, update_delay_s = 0.1):
        for _ in range(self.max_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            #print(np.any(self.obs[0, :,:,3] == 2 * 85))
            #print(actions)
            print(self.env.venv.envs[0].game)
            sleep(update_delay_s)

def main():
    parser = argparse.ArgumentParser(
        description="play snake automatically using network")
    parser.add_argument(
        "--model_file",
        help="the model file to use",
        type=str,
        default="ppo_model")
    parser.add_argument(
        "--policy", help="one of cnn, lst and lnlstm", type=str, default="cnn")

    args = parser.parse_args()
    play_snake(args.model_file, args.policy)


if __name__ == "__main__":
    main()
