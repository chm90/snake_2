#!/usr/bin/env python3
import sys
import argparse
from baselines import bench, logger
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from random import randint
import numpy as np

def train(num_timesteps, seed, policy,load_path):
    from baselines.common import set_global_seeds
    from baselines.common.atari_wrappers import wrap_deepmind
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.common.vec_env.vec_frame_stack import VecFrameStack
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
    from snake_gym import make_snake
    #import logging
    import multiprocessing
    #import os.path as osp
    import tensorflow as tf
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    #gym.logger.setLevel(logging.WARN)
    tf.Session(config=config).__enter__()

    def make_env(rank):
        def env_fn():
            env = make_snake(shape=(5,5))
            env.seed(randint(0,1000000000))
            env = bench.Monitor(env, logger.get_dir())
            #print("---------------------------------------------------------------")
            #print(obs.shape)
            #print("---------------------------------------------------------------")
            return env
        return env_fn
    nenvs = 16
    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    set_global_seeds(seed)
    env = VecFrameStack(env, 4) #Potantialy required
    policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy, "mlp": MlpPolicy}[policy]
    ppo2.learn(policy=MyPolicy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1),
        load_path=load_path,save_interval=20)

class MyPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            #h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            #h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            #h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(tf.cast(X, tf.float32)/255)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x, init_scale=0.01)
            vf = fc(h4, 'v', 1, act=lambda x:x)[:,0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm',"mlp"], default='cnn')
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--load-path',type=str,default=None)
    args = parser.parse_args()
    #logger.configure()
    train(num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy,load_path = args.load_path)

if __name__ == '__main__':
    main()