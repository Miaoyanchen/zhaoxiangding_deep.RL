---
title: "A Comprehensive Introduction to Deep Reinforcement Learning"
author: "Miaoyan Chen and Zhaoxiang Ding"
date: "2024-05-07"
output:
  bookdown::html_document2:
    toc: true
    toc_float: true
    toc_depth: 3
    number_sections: true
    theme: united
    highlight: tango
    code_folding: show
    code_download: true
    fig_caption: true
    keep_md: true
bibliography: deepRL.references.bib
---



## Table of Contents

1.  Preamble

2.  Summary of Notations

3.  Reinforcement Learning

4.  Deep Reinforcement Learning

# Preamble

Whether we are engaging in a daily conversation or to driving a vehicle, essentially we are picking up on our surrounding environment and making a response simultaneously to the changes in the environment. Essentially, we are learning from our interaction with the environment and executing an action in return. This is the foundation idea that lies in all learning and intelligence [@sutton2018]. A machine learning technique to train software to make decision to achieve the optimal result by performing a sequence of actions in an environment is known as *reinforcement learning* [citation]. A reinforcement learning environment is formalized by an optimal control of Markov decision processes, which can be decompose into three essential parts - sensation, action, and goal [@sutton2018]. A learning agent should be able to sense the state of the environment, and takes series of actions that effects the state and achieve a goal overtime.

The idea of reinforcement learning is further extended to *deep reinforcement learning* that handles more sophisticated tasks. Deep reinforcement learning allows the agent to perform on real-world complexity in higher dimensions by training through deep neural networks. A novel artificial agent, deep Q-network agent was first introduced by Mnih et al. (2015).

# Summary of Notations

| Variable   | Definition                                                  |
|------------|-------------------------------------------------------------|
| $s$        | state                                                       |
| $a$        | action                                                      |
| $t$        | discrete time step                                          |
| $\pi$      | policy, decision rule                                       |
|            |                                                             |
| $s_t$      | state at time $t$                                           |
| $a_t$      | game actions selected from a set of actions                 |
| $x_t$      | result form emulator in the form of vector of pixel values  |
| $r_t$      | reward in game                                              |
| $\gamma$   | discount rate parameter per time step (default set to 0.99) |
| $Q^*(s,a)$ | maximum expected return achieved by policy                  |
| $\theta$   | weight parameter                                            |
| $R_t$      | total reward at time t, dependent                           |

# Reinforcement Learning

## Q-learning Algorithm

In order to teach the agent to achieve the maximum reward, and tie the reward to the actions, we can define the **Optimal action-value function**:
$$
Q^*(s,a) = \max_{\pi} \mathbb{E}[R_t | s_t = s, a_t = a, \pi]
$$
Where, $\pi$ is the policy to guide the action, $s$ is the current state, $a$ is the current action, and $R_t$ is the total reward at time $t$.

This function obey *Bellman equation*, which is based on the following intuition: the expected return for taking the optimal action from a given state is the sum of the immediate reward from the current state to the next state, and the expected return from the next state to the goal state. Following this equation, The optimal action-value function can be expressed as:

$$
Q^*(s,a) = \mathbb{E}_{s'}[r + \gamma \max_{a'} Q^*(s',a') | s,a]
$$
Where $r$ is the current reward, $\gamma$ is the discount rate parameter, $s'$ is the next state and $a'$ is the next action.

Fig \@ref(fig:qvalue) shows a visualization of the learned action-value function on the game Pong. Pong is a two-player game where the goal is to hit the ball past the opponent's paddle. The agent controls one of the paddles (green one) and the opponent is controlled by a simple AI. At time point 1, the ball is moving towards the paddle controlled by the agent on the right side of the screen and the q-values of all actions are around 0.7, reflecting the expected value of this state based on previous experience. At time point 2, the agent starts moving the paddle towards the ball and the value of the ‘up’ action stays high while the value of the ‘down’ action falls to −0.9. This reflects the fact that pressing ‘down’ would lead to the agent losing the ball and incurring a reward of −1 (loose the game). At time point 3, the agent hits the ball by pressing ‘up’ and the expected reward keeps increasing until time point 4, when the ball reaches the left edge of the screen and the value of all actions reflects that the agent is about to receive a reward of 1. Note, the dashed line shows the past trajectory of the ball purely for illustrative purposes (that is, not shown during the game).

<div class="figure" style="text-align: center">
<img src="figure-qvalue.png" alt="Visualization if the learned action-value function on the game Pong" width="2598" />
<p class="caption">(\#fig:qvalue)Visualization if the learned action-value function on the game Pong</p>
</div>

## Q-network: approximator of Q-value function



## Deep Reinforcement Learning

### A Novel Artificial Agent - Deep Q network (DQN)

### References

::: {#refs}
:::
