# 20200419
import datetime
import os
import sys

import gym
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os.path as osp
import time
from baselines import logger
from runner import Runner_svib
# from baselines.a2c.a2c import Runner
from baselines.a2c.utils import conv, lstm, lnlstm, conv_to_fc, fc, ortho_init, cat_entropy, find_trainable_variables, Scheduler, make_path, batch_to_seq, seq_to_batch
from baselines.common import set_global_seeds, tf_util
from baselines.common.cmd_util import make_atari_env
from baselines.common.distributions import make_pdtype
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat
from gym.wrappers import Monitor
from tensorflow import sqrt
from tensorflow.contrib.solvers.python.ops.util import l2norm

from utils import grad_clip, batch_norm, explained_variance, generate_noise, tf_l2norm, mse
from policy import CnnPolicySVIB

start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
env_name = "MoveToBeaconNoFrameskip-v1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#!/usr/bin/env Python
# coding=utf-8

class Model_A2C_SVIB(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, master_ts = 1, worker_ts = 8,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=2.5, lr=7e-4, cell=256,
                 ib_alpha=0.04, sv_M=32, algo='use_svib_uniform',
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.make_session()
        nact = ac_space.n
        nbatch = nenvs*master_ts*worker_ts # master what's mean?

        # A:action, ADV:advantage, R:reward, LR:Learning Rate
        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, 1, cell=cell, M=sv_M, model='step_model', algo=algo)
        train_model = policy(sess, ob_space, ac_space, nbatch, master_ts, worker_ts, cell = cell, M=sv_M, model='train_model', algo=algo)
        print('model_setting_done, algorithm:', str(algo))

        '''
        可视化互信息，暂时跳过
        '''
        ib_loss = train_model.mi_xh_loss
        T = train_model.T_value
        t_grads, t_global_norm = grad_clip(-vf_coef*ib_loss, max_grad_norm, ['model/T/update_params'])
        t_trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _t_train = t_trainer.apply_gradients(t_grads)
        T_update_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/T/update_params')
        T_orig_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/T/orig_params')
        reset_update_params = [update_param.assign(orig_param) for update_param, orig_param in zip(T_update_params, T_orig_params)]

        # rpf_matrix, rpf_grads = rpf_kernel(vf_loss_sv, rpf_h)

        if algo == 'use_svib_uniform' or algo == 'use_svib_gaussian':
            def expand_placeholder(X, M=sv_M):
                return tf.tile(tf.expand_dims(X, axis=-1), [1, M])
            A_expand, R_expand = expand_placeholder(A), expand_placeholder(R)
            neglogpac_expand = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.wpi_expand, labels=A_expand)#shape=[nbatch, sv_M]
            # pg_loss_expand = tf.reduce_mean(ADV_expand * neglogpac_expand, axis=-1)
            pg_loss_expand = tf.reduce_mean(tf.stop_gradient(R_expand-train_model.wvf_expand[:,:,0]) * neglogpac_expand, axis=-1)
            vf_loss_expand = tf.reduce_mean(mse(tf.squeeze(train_model.wvf_expand), R_expand), axis=-1)
            entropy_expand = tf.reduce_mean(cat_entropy(train_model.wpi_expand), axis=-1)#shape=[nbatch]
            J_theta = -(pg_loss_expand + vf_coef*vf_loss_expand - ent_coef*entropy_expand)

            loss_expand = -J_theta / float(nbatch)
            pg_loss_expand_ = tf.reduce_mean(pg_loss_expand)
            vf_loss_expand_ = tf.reduce_mean(vf_loss_expand)
            entropy_expand_ = tf.reduce_mean(entropy_expand)
            loss_expand_ = -tf.reduce_mean(J_theta)

            print('ib_alpha: ', ib_alpha)
            log_p_grads = tf.gradients(J_theta/np.sqrt(ib_alpha), [train_model.wh_expand])[0]#shape=[nbatch, sv_M, cell]
            if algo == 'use_svib_gaussian':
                mean, var = tf.nn.moments(train_model.wh_expand, axes=1, keep_dims=True)#shape=[nbatch, 1,cell]
                gaussian_grad = -(train_model.wh_expand - mean)/(float(sv_M) * (var+1e-3))
                log_p_grads += 5e-3*(tf_l2norm(log_p_grads, axis=-1, keep_dims=True)/tf_l2norm(gaussian_grad, axis=-1, keep_dims=True))*gaussian_grad
            sv_grads = tf.constant(0., tf.float32, shape=[nbatch, 0, cell])
            for i in range(sv_M):
                sv_grad = tf.reduce_sum(train_model.rpf_matrix[:, :, i:i+1] * log_p_grads, axis=1) + np.sqrt(ib_alpha)*train_model.rpf_grads[:, i, :]#shape=[nbatch, cell]
                sv_grads = tf.concat([sv_grads, tf.expand_dims(sv_grad, axis=1)], axis=1)
                
            SV_GRADS = tf.placeholder(tf.float32, [nbatch, sv_M, cell])
            repr_loss = tf.reduce_mean(SV_GRADS * train_model.wh_expand, axis=1)#shape=[nbatch,cell]
            repr_loss = -tf.reduce_mean(tf.reduce_sum(repr_loss, axis=-1))#max optimization problem to minimization problem
            # repr_loss = -tf.reduce_mean(repr_loss, axis=0)

            # sv_grad_ = tf.reduce_sum(train_model.rpf_matrix[:, :, 2:3] * log_p_grads, axis=1) + train_model.rpf_grads[:, 2, :]
            # exploit_term = tf.reduce_sum(train_model.rpf_matrix[:, :, 2:3] * log_p_grads, axis=1)
            # explore_term = train_model.rpf_grads[:, 2, :]
            grads_expand, global_norm_expand = grad_clip(loss_expand, max_grad_norm, ['model/worker_module'])
            trainer_expand = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
            _train_expand = trainer_expand.apply_gradients(grads_expand)

            repr_grads, repr_global_norm = grad_clip(repr_loss, max_grad_norm, ['model/ordinary_encoder'])
            repr_trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
            _repr_train = repr_trainer.apply_gradients(repr_grads)

        elif algo == 'sv_a2c':
            def expand_placeholder(X, M=sv_M):
                return tf.tile(tf.expand_dims(X, axis=-1), [1, M])
            A_expand, R_expand = expand_placeholder(A), expand_placeholder(R) # [40, 32]
            sigma = tf.constant(1e-5)
            neglogpac_expand = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.wpi_expand, labels=A_expand) + sigma # [40, 32]
            pg_loss_expand = tf.reduce_mean(tf.stop_gradient(R_expand - train_model.wvf_expand[:, :, 0]) * neglogpac_expand, axis=-1) # [40, ]
            vf_loss_sv = tf.expand_dims(mse(tf.squeeze(train_model.wvf_expand), R_expand), axis=-1) # [40, 32, 1]
            vf_loss_expand = tf.reduce_mean(mse(tf.squeeze(train_model.wvf_expand), R_expand), axis=-1) # [40, ]
            entropy_expand = tf.reduce_mean(cat_entropy(train_model.wpi_expand), axis=-1)  # shape=[nbatch]

            J_theta = pg_loss_expand + vf_coef * vf_loss_expand - ent_coef * entropy_expand # [40, ]
            # 为什么要除nbatch
            loss_expand = J_theta / float(nbatch) # [40, ]

            pg_loss_expand_ = tf.reduce_mean(pg_loss_expand)
            vf_loss_expand_ = tf.reduce_mean(vf_loss_expand) # [1]
            entropy_expand_ = tf.reduce_mean(entropy_expand)
            loss_expand_ = tf.reduce_mean(J_theta)

            print('ib_alpha: ', ib_alpha)
            # mean, var = tf.constant(0., tf.float32, [nbatch, 1, 1]), tf.constant(1, tf.float32, [nbatch, 1, 1])
            mean, var = tf.nn.moments(vf_loss_sv, axes=1, keep_dims=True) # [40, 1, 1]
            # Problem1: guassian gradient cauculate problem
            log_p_grads = -(vf_loss_sv - mean) / (float(sv_M) * (var))

            sv_grads = tf.constant(0., tf.float32, shape=[nbatch, 0, 1]) # [nbatch, m, 1]

            rpf_h = self.h_coef(vf_loss_sv, sv_M)
            rpf_matrix, rpf_grads = self.rpf_kernel(vf_loss_sv, rpf_h, sv_M)
            for i in range(sv_M):
                # sv_grad = tf.reduce_sum(train_model.rpf_matrix[:, :, i:i+1] * log_p_grads, axis=1) + sqrt(ib_alpha) * train_model.rpf_grads[:, i, :] #shape=[nbatch, cell]
                sv_grad = tf.reduce_sum(rpf_matrix[:, :, i:i + 1] * log_p_grads, axis=1) + rpf_grads[:, i, :]
                sv_grads = tf.concat([sv_grads, tf.expand_dims(sv_grad, axis=1)], axis=1)

            SV_GRADS = tf.placeholder(tf.float32, [nbatch, sv_M, 1])
            sv_loss = tf.reduce_mean(SV_GRADS * vf_loss_sv, axis=1)

            loss_expand -=  ib_alpha * (tf_l2norm(loss_expand, axis=-1, keep_dims=True)/tf_l2norm(sv_loss, axis=-1, keep_dims=True)) * sv_loss

            grads_expand, global_norm_expand = grad_clip(loss_expand, max_grad_norm, ['model/worker_module', 'model/ordinary_encoder'])
            trainer_expand = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
            _train_expand = trainer_expand.apply_gradients(grads_expand)

            # sv_loss_grads, sv_global_norm = grad_clip(sv_loss, max_grad_norm, ['model/worker_module/comm', 'model/worker_module/w_value'])
            # sv_trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
            # _sv_train = sv_trainer.apply_gradients(sv_loss_grads)


        elif algo == 'anchor':
            neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.wpi, labels=A)
            pg_loss = tf.reduce_mean(ADV * neglogpac)
            entropy = tf.reduce_mean(cat_entropy(train_model.wpi))
            vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.wvf), R))

            # anchor method
            param_list = []
            for scope in ['model/worker_module']:
                List = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                print(len(List))
                param_list += List

            param_value_layer1_w = param_list[0]
            param_value_layer1_b = param_list[1]
            param_policy_layer2_w = param_list[2]
            param_value_layer2_w = param_list[4]
            param_value_layer2_b = param_list[5]

            init_stddev = 5.0 # 7.0
            init_stddev_2 = 0.18 / np.sqrt(cell)  # normal scaling
            lambda_anchor = [0.000001,0.1]

            layer1_w_init = tf.random_normal(mean=0., stddev=init_stddev, shape=param_value_layer1_w.get_shape())
            layer1_b_init = tf.random_normal(mean=0., stddev=init_stddev, shape=param_value_layer1_b.get_shape())
            layer2_w_init = tf.random_normal(mean=0, stddev=init_stddev_2, shape=param_value_layer2_w.get_shape())
            layer2_b_init = tf.random_normal(mean=0, stddev=init_stddev_2, shape=param_value_layer2_b.get_shape())

            loss_anchor = lambda_anchor[0] / nbatch * tf.reduce_sum(tf.square(layer1_w_init - param_value_layer1_w))
            loss_anchor += lambda_anchor[0] / nbatch * tf.reduce_sum(tf.square(layer1_b_init - param_value_layer1_b))
            loss_anchor += lambda_anchor[1] / nbatch * tf.reduce_sum(tf.square(layer2_w_init - param_value_layer2_w))
            loss_anchor += lambda_anchor[1] / nbatch * tf.reduce_sum(tf.square(layer2_b_init - param_value_layer2_b))

            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy + loss_anchor

            grads, global_norm = grad_clip(loss, max_grad_norm, ['model/worker_module', 'model/ordinary_encoder'])
            trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
            _train = trainer.apply_gradients(grads)

        else: # regular algorithm
            neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.wpi, labels=A)
            pg_loss = tf.reduce_mean(ADV * neglogpac)
            entropy = tf.reduce_mean(cat_entropy(train_model.wpi))
            vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.wvf), R))

            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy

            grads, global_norm = grad_clip(loss, max_grad_norm, ['model/worker_module', 'model/ordinary_encoder'])
            trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
            _train = trainer.apply_gradients(grads)

        params = find_trainable_variables("model")
        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)


        def train(wobs, whs, states, rewards, masks, actions, values, noises):
            advs = rewards - values
            # adv_mu, adv_var = np.mean(advs), np.var(advs)+1e-3
            # advs = (advs - adv_mu) / adv_var

            for step in range(len(whs)):
                cur_lr = lr.value()
            sv_td_map = {train_model.wX : wobs, train_model.istraining:True, A:actions, R:rewards, LR:cur_lr}

            # Sess Graph
            # writer = tf.summary.FileWriter('./', sess.graph)

            repr_td_map = {train_model.wX: wobs, train_model.istraining: True, A: actions, R: rewards, LR: cur_lr}
            rl_td_map = {train_model.wX : wobs, train_model.istraining: True, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                rl_td_map[train_model.wS] = states
                rl_td_map[train_model.wM] = masks
            repr_grad_norm = 0.
            # print(str(np.sum(whs-sess.run(train_model.wh, feed_dict={train_model.wX : wobs, train_model.istraining:True, train_model.noise:noises}))))
            if algo == 'use_svib_uniform' or algo == 'use_svib_gaussian':
                repr_td_map[train_model.noise_expand], repr_td_map[train_model.NOISE_KEEP] = sess.run(train_model.noise_expand), noises
                wh_expands, sv_gradients = sess.run([train_model.wh_expand, sv_grads], feed_dict=repr_td_map)
                rl_td_map[train_model.wh_expand] = wh_expands
                tloss, value_loss, policy_loss, policy_entropy, rl_grad_norm, _ = sess.run(
                    [loss_expand_, vf_loss_expand_, pg_loss_expand_, entropy_expand_, global_norm_expand, _train_expand],
                    feed_dict=rl_td_map
                )
                repr_td_map[SV_GRADS] = sv_gradients
                # if algo == 'use_svib_gaussian':
                #     gaussian_gradients, repr_grad_norm, __ =\
                #         sess.run([gaussian_grad, repr_global_norm, _repr_train], feed_dict=repr_td_map)
                #     return tloss, value_loss, policy_loss, policy_entropy, rl_grad_norm, gaussian_gradients, repr_grad_norm  # represnet_loss, SV_GRAD, EXPLOIT, LOG_P_GRADS, EXPLORE
                repr_grad_norm, represent_loss, __ = sess.run([repr_global_norm, repr_loss, _repr_train], feed_dict=repr_td_map)

            elif algo == 'anchor':
                rl_td_map[train_model.wX] = wobs
                tloss, value_loss, policy_loss, policy_entropy, rl_grad_norm, anchor_loss, _ = sess.run(
                    [loss, vf_loss, pg_loss, entropy, global_norm, loss_anchor ,_train],
                    feed_dict=rl_td_map
                )

                represent_loss = 0.
                sv_loss_ = 0
            elif algo == 'sv_a2c':
                sv_td_map[train_model.noise_expand], sv_td_map[train_model.NOISE_KEEP] = sess.run(
                    train_model.noise_expand), noises
                wvf_expands, sv_gradients = sess.run([train_model.wvf_expand, sv_grads], feed_dict=sv_td_map)
                rl_td_map[train_model.wvf_expand] = wvf_expands
                rl_td_map[train_model.noise_expand], rl_td_map[train_model.NOISE_KEEP] = sess.run(
                    train_model.noise_expand), noises
                rl_td_map[SV_GRADS] = sv_gradients
                tloss, value_loss, policy_loss, policy_entropy, rl_grad_norm, sv_loss_, _ = sess.run(
                    [loss_expand_, vf_loss_expand_, pg_loss_expand_, entropy_expand_, global_norm_expand,
                     sv_loss, _train_expand],
                    feed_dict=rl_td_map
                )
                sv_td_map[SV_GRADS] = sv_gradients
                anchor_loss = 0.
                represent_loss = 0.

            else:
                rl_td_map[train_model.wX], rl_td_map[train_model.noise] = wobs, noises#noise won't be used when algo is 'regular'
                tloss, value_loss, policy_loss, policy_entropy, rl_grad_norm, _ = sess.run(
                    [loss, vf_loss, pg_loss, entropy, global_norm, _train],
                    feed_dict=rl_td_map
                )
                # repr_td_map[WH_GRADS] = wh_gradients
                # repr_grad_norm, __ = sess.run([ordin_repr_global_norm, _ordin_repr_train], feed_dict=repr_td_map)
                repr_grad_norm = 0.
                represent_loss = 0.
                anchor_loss = 0
                sv_loss_ = 0
            return tloss, value_loss, policy_loss, policy_entropy, rl_grad_norm, repr_grad_norm, represent_loss, anchor_loss, sv_loss_#SV_GRAD, EXPLOIT, LOG_P_GRADS, EXPLORE

        def train_mine(wobs, whs, steps=256, lr=7e-4):
            # whs_std = (whs-np.mean(whs,axis=0,keepdims=True))/(1e-8 + np.std(whs,axis=0,keepdims=True))
            idx = np.arange(len(whs))
            ___ = sess.run(reset_update_params)
            for i in range(int(steps)):
                np.random.shuffle(idx)
                mi, T_value, __ = sess.run([ib_loss, T, _t_train],
                                           feed_dict={train_model.wX: wobs[idx], train_model.wh: whs[idx],
                                                      LR: lr, train_model.istraining: True})
            logger.record_tabular('mutual_info_loss', float(mi))
            logger.record_tabular('T_value', float(T_value))
            logger.dump_tabular()

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_mine = train_mine
        self.train_model = train_model
        self.step_model = step_model
        self.get_wh = step_model.get_wh
        self.get_noise = step_model.get_noise
        self.value = step_model.wvalue
        self.step = step_model.step
        self.initial_state = step_model.w_initial_state
        self.save = save
        self.load = load
        self.sv_M = sv_M
        # self.rpf_h = rpf_h
        # self.rpf_matrix = rpf_matrix
        # self.rpf_grads = rpf_grads
        tf.global_variables_initializer().run(session=sess)

    def h_coef(self, Z_expand, M=32):
        '''
        :param Z_expand: shape = [nbatch, M, cell]
        :param M:
        :return: h coefficients of RPF kernel
        '''
        rpf_hs = tf.constant(0., dtype=tf.float32, shape=[Z_expand.get_shape()[0].value, 0]) # [40, 0] Z_expand=[40, 32]
        for i in range(M - 1):
            for j in range(i + 1, M):
                distance = tf.reduce_sum(tf.square(Z_expand[:, i, :] - Z_expand[:, j, :]), axis=1,
                                         keep_dims=True)  # shape=[nbatch, 1]
                rpf_hs = tf.concat([rpf_hs, distance], axis=1)
        values, idx = tf.nn.top_k(rpf_hs, k=rpf_hs.get_shape()[-1].value // 2)  # shape=[nbatch, M(M-1)/4]
        rpf_h = values[:, -1]  # shape=[nbatch]
        return tf.stop_gradient(0.5 * rpf_h / tf.log(float(M) + 1.))  # prevent denominator to be 0

    def rpf_kernel(self, Z_expand, rpf_h, M=32):
        '''
        :param Z_expand: shape=[nbatch, M, cell]
        :param Z_expand_repr_train: shape=[nbatch, M, cell]
        :param rpf_h:
        :return: a [nbatch, M, M] kernel matrix and its gradients towards Z_expand_column_dim
        '''
        Z_expand_row_dim = tf.expand_dims(Z_expand, axis=1)  # shape=[nbatch,1,M,cell]
        Z_expand_column_dim = tf.expand_dims(Z_expand, axis=2)  # shape=[nbatch,M,1,cell]
        delta = Z_expand_column_dim - Z_expand_row_dim  # shape=[nbatch, M, M, cell]
        delta_square = tf.reduce_sum(tf.square(delta), axis=-1)  # shape=[nbatch, M, M]
        rpf_h_expand = tf.reshape(rpf_h, [-1, 1, 1])
        rpf_matrix = tf.exp(-delta_square / rpf_h_expand)  # shape=[nbatch, M, M]
        rpf_grads = tf.constant(0., dtype=tf.float32,
                                shape=[Z_expand.get_shape()[0].value, 0, Z_expand.get_shape()[2].value])
        for i in range(M):
            rpf_grad = tf.reduce_mean(tf.gradients(rpf_matrix[:, :, i], [Z_expand_column_dim])[0], axis=1)
            rpf_grads = tf.concat([rpf_grads, rpf_grad], axis=1)
        return tf.stop_gradient(rpf_matrix), tf.stop_gradient(rpf_grads)  # rpf_grads.shape=[nbatch, M, cell]

class test_Runner(object):
    def __init__(self, env, model, num_episodes=10):
        self.env = env
        self.model = model
        self.num_episodes = num_episodes
        obs = env.reset()
        self.obs = np.copy(obs)
        self.states = model.initial_state
        self.done = False
        self.dones = [False for _ in range(env.num_envs)]
        self.one_episode_r = 0.
        self.total_r = 0.

    def run(self):
        self.one_episode_r = 0.
        self.total_r = 0.
        for i in range(self.num_episodes):
            while not self.done:
                noises, actions, values, self.states, _, h = self.model.step(self.obs, self.states, self.dones)
                self.obs, rewards, self.dones, _ = self.env.step(actions)
                self.done = self.dones[0]
                self.one_episode_r += rewards
            self.total_r += self.one_episode_r
            self.one_episode_r = 0.
            self.done = False
        return (self.total_r[0] / float(self.num_episodes))

def learn(policy, env, test_env, seed, master_ts = 1, worker_ts=8, cell = 256,
          ent_coef = 0.01, vf_coef = 0.5, max_grad_norm = 2.5, lr = 7e-4,
          alpha = 0.99, epsilon = 1e-5, total_timesteps = int(80e6), lrschedule = 'linear',
          ib_alpha = 1e-3, sv_M = 32, algo='use_svib_uniform',
          log_interval = 10, gamma = 0.99, load_path="saved_nets-data/hrl_a2c/%s/data"%start_time):

    tf.reset_default_graph()
    set_global_seeds(seed)
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    print(str(nenvs)+"------------"+str(ob_space)+"-----------"+str(ac_space))
    model = Model_A2C_SVIB(policy = policy, ob_space = ob_space, ac_space = ac_space, nenvs = nenvs, master_ts=master_ts, worker_ts=worker_ts,
                           ent_coef = ent_coef, vf_coef = vf_coef, max_grad_norm = max_grad_norm, lr = lr, cell = cell,
                           ib_alpha = ib_alpha, sv_M = sv_M, algo=algo,
                           alpha = alpha, epsilon = epsilon, total_timesteps = total_timesteps, lrschedule = lrschedule)
    try:
        model.load(load_path)
    except Exception as e:
        print("no data to load!!"+str(e))
    runner = Runner_svib(env = env, model = model, nsteps=master_ts*worker_ts, gamma=gamma)
    test_runner = test_Runner(env = test_env, model = model)

    tf.get_default_graph().finalize()
    nbatch = nenvs * master_ts * worker_ts
    tstart = time.time()
    reward_list = []
    value_list = []
    prev_r = 0.
    exp_coef = .1
    for update in range(1, total_timesteps//nbatch+1):
        b_obs, b_whs, states, b_rewards, b_wmasks, b_actions, b_values, b_noises = runner.run()
        # print(np.max(b_obs[0]))
        # if algo == 'use_svib_gaussian':
        #     tloss, value_loss, policy_loss, policy_entropy, rl_grad_norm, gaussian_gradients, repr_grad_norm =\
        #         model.train(b_obs, b_whs, states, b_rewards, b_wmasks, b_actions, b_values)
        #     # print(gaussian_gradients[0, 3:5, 0:30])
        #     # print('1')
        # else:
        tloss, value_loss, policy_loss, policy_entropy, rl_grad_norm, repr_grad_norm, represent_loss, anchor_loss, sv_loss = \
            model.train(b_obs, b_whs, states, b_rewards, b_wmasks, b_actions, b_values, b_noises)
        # print('b_whs:', b_whs[0, 0:60])
        # print('sv_grad:',SV_GRAD[0, 0:40])
        # print('exploit:',EXPLOIT[0, 0:40])
        # print('log_p_grads:',LOG_P_GRADS[0, 3:5, 0:40])
        # print('explore:',EXPLORE[0, 0:40])
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(b_values, b_rewards)
            logger.record_tabular("fps", fps)
            logger.record_tabular("tloss", float(tloss))
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular('repr_grad_norm', float(repr_grad_norm))
            logger.record_tabular('rl_grad_norm', float(rl_grad_norm))
            logger.record_tabular('repr_loss', float(represent_loss))
            logger.record_tabular('anchor_loss',float(anchor_loss))
            logger.record_tabular('sv_loss',float(np.mean(sv_loss)))
            # if algo == 'use_svib_gaussian':
            #     logger.record_tabular('gaussian_grad_norm_without_clip', float(np.mean(np.abs(gaussian_gradients[0]))))
            # logger.record_tabular('represent_loss', float(represent_loss))
            logger.record_tabular('represent_mean', float(np.mean(b_whs[0])))
            logger.dump_tabular()
            if update % (200*log_interval) == 0 or update == 1:
                save_th = update//(200*log_interval)
                model.save("saved_nets-data/%s/hrl_a2c_svib/%s/%s/data" % (env_name, start_time, save_th))
                # model.train_mine(b_obs, b_whs)
                # episode_r = exp_coef*(test_runner.run()) + (1-exp_coef)*prev_r
                episode_r = exp_coef*(test_runner.run()) + (1-exp_coef)*prev_r
                prev_r = np.copy(episode_r)
                reward_list.append(episode_r)
                value_list.append(value_loss)
                logger.record_tabular('episode_r', float(episode_r))
                logger.dump_tabular()
    env.close()
    tf.reset_default_graph()
    return reward_list, value_list

def make_atari_low_dim(env_id):
    env = gym.make(env_id)
    # print(env.observation_space.shape)
    return env

def make_atari_env_low_dim(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari_low_dim(env_id)
            env.seed(seed + rank)
            # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def train(env_id, num_timesteps, seed, policy, lrschedule, num_env,load_path,
          algo='use_svib_uniform', ib_alpha=1e-3):
    if policy == 'cnn_svib':
        policy_fn = CnnPolicySVIB
    else:
        policy_fn = CnnPolicySVIB
    if 'NoFrameskip' in env_id:
        env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
        test_env = VecFrameStack(make_atari_env(env_id, num_env, seed+1), 4)
    else:
        env = VecFrameStack(make_atari_env_low_dim(env_id, num_env, seed), 4)
        test_env = VecFrameStack(make_atari_env_low_dim(env_id, num_env, seed+1), 4)
        # train_mine_env = VecFrameStack(make_atari_env_low_dim(env_id, num_env, seed), 4)
    reward_list, value_list = learn(policy_fn, env, test_env, seed, total_timesteps=int(num_timesteps),
                        lrschedule=lrschedule, load_path=load_path, algo=algo, ib_alpha=ib_alpha)
    env.close()
    return reward_list, value_list

def config_log(FLAGS):
    logdir = "tensorboard/%s/hrl_a2c_svib/%s_lr%s_%s/%s_%s_%s_%s" % (
        FLAGS.env,FLAGS.num_timesteps, '0.0007',FLAGS.policy,start_time, FLAGS.env, "anchor", FLAGS.ib_alpha)
    if FLAGS.log == "tensorboard":
        Logger.DEFAULT = Logger.CURRENT = Logger(dir=logdir, output_formats=[TensorBoardOutputFormat(logdir)])
    elif FLAGS.log == "stdout":
        Logger.DEFAULT = Logger.CURRENT = Logger(dir=logdir, output_formats=[HumanOutputFormat(sys.stdout)])

def train_all(algos, env_id, num_timesteps, seed, policy, lrschedule, num_env, load_path, great_time):
    """
    :param algos:
    :param env_id:
    :param num_timesteps:
    :param seed: number of seed in i, change the number of i. if want to use multiseed algorithm
    :param policy:
    :param lrschedule:
    :param num_env:
    :param load_path:
    :return:
    """
    all = pd.DataFrame([])

    # 换成multi-seed的时候小心一下value_list
    for i in range(3):
        for algo in algos.keys():
            ib_alpha = algos[algo]
            # 换成一个列向量，(n,1)
            reward_list_, value_list_ = train(env_id, num_timesteps=num_timesteps, seed=seed+i, policy=policy, lrschedule=lrschedule,
                                              num_env=num_env, load_path=load_path, algo=algo, ib_alpha=ib_alpha)

            reward_list = np.reshape(np.array(reward_list_), (-1, 1))
            value_list = np.reshape(np.array(value_list_), (-1, 1))
            len = reward_list.shape[0]
            data = pd.DataFrame(np.ones((len, 5)))
            data.columns = ['step', 'avg_reward', 'algorithm', 'seed', 'value_list']
            data['step'] = np.linspace(0, len-1, len).reshape(-1, 1)
            data['avg_reward'] = reward_list
            data['algorithm'] = algo
            data['seed'] = seed+i
            data['value_list'] = value_list
            all = pd.concat([all, data], 0)
    sns.lineplot(x='step', y='avg_reward', data=all, hue='algorithm')
    #　the final figure of all the experiment result
    plt.savefig("./result_image/reward/%s_%s.png" % (env_id, start_time))
    plt.show()
    plt.clf()
    sns.lineplot(x='step', y='value_list', data=all, hue='algorithm')
    plt.savefig("./result_image/bellman_error/%s_%s.png" % (env_id, start_time))

    return all

def main():
    global env_name
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='AssaultNoFrameskip-v4') # BeamRiderNoFrameskip-v4, PongNoFrameskip-v4
    parser.add_argument('--num_env', help='number of environments', type=int, default=5)
    parser.add_argument('--seed', help='RNG seed', type=int, default=42)
    parser.add_argument('--num_timesteps', type=int, default=int(14e6))
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn_svib', 'lstm_svib'], default='cnn_svib')
    parser.add_argument('--train_option', help='which algorithm do we train',
                        choices=['compare_with_regular', 'compare_with_none', 'uniform', 'gaussian', 'regular', 'regular_with_noise'],
                        default='compare_with_regular')
    #　compare_with_regular: 2 SVIB algorithms and 2 regular A2C algorithms; compare_with_none: 2 SVIB algorithms
    # uniform: SVIB with uniform distribution as prior distribution; gaussian: SVIB with Gaussian distributions
    # regular: A2C ; regular_with_noise: A2C with additional noise
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear', 'double_linear_con'], default='double_linear_con')
    parser.add_argument('--log', help='logging type', choices=['tensorboard', 'stdout'], default='tensorboard')
    parser.add_argument('--great_time', help='the time gets great result:%Y%m%d%H%M', default=202008291101)#default='201904181134')
    parser.add_argument('--great_th', help='the timeth gets great result', default=45)
    parser.add_argument('--ib_alpha', type=float, default=0.11)
    args = parser.parse_args()
    config_log(args)
    env_name = args.env
    print(args.ib_alpha)

    load_path = "saved_nets-data/%s/hrl_a2c_svib/%s/%s/data" % (args.env, args.great_time,args.great_th)
    if args.train_option == 'compare_with_regular':
        # algos = {'regular': args.ib_alpha,'use_svib_uniform': args.ib_alpha, 'regular_with_noise': args.ib_alpha,  'use_svib_gaussian': args.ib_alpha,}
        #'sv_a2c': args.ib_alpha, 'sv_a2c':args.ib_alpha,

        # algos = {'sv_gibbs':args.ib_alpha, 'regular': args.ib_alpha}
        algos = {'regular': args.ib_alpha}
        all = train_all(algos=algos, env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                        policy=args.policy, lrschedule=args.lrschedule, num_env=args.num_env, load_path=load_path, great_time = args.great_time)
    elif args.train_option == 'compare_with_none':
        algos = {'use_svib_gaussian': args.ib_alpha, 'use_svib_uniform': args.ib_alpha}
        all = train_all(algos=algos, env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed,
                        policy=args.policy, lrschedule=args.lrschedule, num_env=args.num_env, load_path=load_path)
    elif args.train_option == 'uniform':
        all = train(env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy,
                    lrschedule=args.lrschedule, num_env=args.num_env, load_path=load_path, algo='use_svib_uniform',
                    ib_alpha=args.ib_alpha)
    elif args.train_option == 'gaussian':
        all = train(env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy,
                    lrschedule=args.lrschedule, num_env=args.num_env, load_path=load_path, algo='use_svib_gaussian',
                    ib_alpha=args.ib_alpha)
    elif args.train_option == 'regular':
        all = train(env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy,
                    lrschedule=args.lrschedule, num_env=args.num_env, load_path=load_path, algo='regular',
                    ib_alpha=args.ib_alpha)
    else:
        all = train(env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy,
                    lrschedule=args.lrschedule, num_env=args.num_env, load_path=load_path, algo='regular_with_noise',
                    ib_alpha=args.ib_alpha)
    return all


if __name__ == "__main__":
    print('hello_world')
    all=main()