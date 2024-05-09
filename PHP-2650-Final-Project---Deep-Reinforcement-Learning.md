---
title: "A Comprehensive Introduction to Deep Reinforcement Learning"
author: "Miaoyan Chen and Zhaoxiang Ding"
date: "2024-05-09"
header-includes:
   - \usepackage{algorithm}
   - \usepackage{algpseudocode}
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



# Preamble

Whether we are engaging in a daily conversation or driving a vehicle, essentially we are picking up on our surrounding environment and making a response simultaneously to the changes in the environment. That is, we are learning from our interaction with the environment and executing an action in return. This is the foundation idea that lies in all learning and intelligence [@sutton2018]. A machine learning technique to train software to make decision to achieve the optimal result by performing a sequence of actions in an environment is known as *reinforcement learning* [@sutton2018]. A reinforcement learning environment is formalized by an optimal control of Markov decision processes, which can be decompose into three essential parts - sensation, action, and goal [@sutton2018]. A learning agent should be able to sense the state of the environment, and takes series of actions that effects the state and achieve a goal overtime.

The idea of reinforcement learning is further extended to *deep reinforcement learning* that handles more sophisticated tasks. Deep reinforcement learning allows the agent to perform on real-world complexity in higher dimensions by combining reinforcement learning with a class of deep neural networks. A novel artificial agent, deep Q-network agent was recently introduced by Mnih et al. (2015). The authors implemented Q-learning with conventional neural networks, and evaluated their DQN agent with Atari 2600 and other game platforms. We will describe the mathematical concepts and learning process in the sections below to give a comprehensive introduction of deep reinforcement learning.

We will begin by introducing the fundamentals of reinforcement learning known as Q-learning, then dive into the deep learning deviation of the Q-learning algorithm, namely the deep Q-learning network [@Mnih2015].

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
<div class="figure" style="text-align: center">
<img src="diagram.png" alt="Visualization of Reinforcement Learning" width="820" />
<p class="caption">(\#fig:unnamed-chunk-1)Visualization of Reinforcement Learning</p>
</div>

## Q-learning Algorithm

A reinforcement learning environment follows a large finite **Markov decision process** (MDP). In reinforcement learning, MDP is a stochastic decision process that is comprised of 4 tuples: finite set of states, finite set of actions, state transition function, and reward function. The MDP also follows a Markov property, assuming the next state and the expected reward at $t+1$ only depends on the state at time $t$ and action at time $t$ and not on any other prior events [@Kaelbling1998]. A **policy** is a behavior of an agent at a given time, and it takes the state as an input and returns action as output: $\pi(s) \rightarrow a$. The goal of the agent is to select actions from a set of actions $\{a_1,\cdots,a_k\}$ in a way that maximizes not only the current reward, but also the future reward. There's a standard assumption in Q-learning that assumes the future rewards are discounted by a factor $\gamma$ (default set to 0.99) per time step, so the future discounted return at time $t$ is defined as:

$$
R_t = \sum^T_{t' = t} \gamma^{t-t'}r_{t'}
$$

where T is the final time step, $r_{t'}$ is the reward at time $t'$, and the term $\gamma^{t-t'}$ diminishes as we increase in time.

In order to teach the agent to achieve the maximum reward, and tie the reward to the actions, we can define the **Optimal action-value function** as the maximum expected return achieved by following some policy, after the agent seeing some sequence of states $s$ and then taking some action $a$:

$$
Q^*(s,a) = \max_{\pi} \mathbb{E}[R_t | s_t = s, a_t = a, \pi]
$$

where, $\pi$ is the policy to guide the action, $s$ is the observed states, $a$ is the current action, and $R_t$ is the total reward at time $t$.

In our case (learning how to play a game in Atari 2600), $s$ can also be viewed as current states (screen shots of the game), because the consequence of previous actions and states are all reflected in the current state. The agent can make the best decision by observing the current state and taking the action that maximize the expected return.

This function obeys the *Bellman equation*, which is based on the following intuition: the expected return for taking the optimal action from a given state is the sum of the immediate reward from the current state to the next state, and the expected return from the next state to the goal state. Following this equation, The optimal action-value function (Q-function) can be expressed as:

$$
Q^*(s,a) = \mathbb{E}_{s'}[r + \gamma \max_{a'} Q^*(s',a') | s,a]
$$ 

where $r$ is the current reward, $\gamma$ is the discount rate parameter, $s'$ is the next state and $a'$ is the next action.

Fig \@ref(fig:qvalue) shows a visualization of the learned action-value function in the game Pong. The top of the image shows the screenshots from Pong, and the bottom of the image depicts the potential action values that can be gained by taking these actions. Pong is a two-player game where the goal is to hit the ball past the opponent's paddle. The DQN agent controls one of the paddles (green one) and the opponent is controlled by a simple AI. We can use the optimal action-value function to determine the optimal policy that our agent can follow. At time point 1, the ball is moving towards the paddle controlled by the agent on the right side of the screen and the q-values of all actions are around 0.7, reflecting the expected value of this state based on previous experience. At time point 2, the agent starts moving the paddle towards the ball and the value of the 'up' action stays high while the value of the 'down' action falls to −0.9. This reflects the fact that pressing 'down' would lead to the agent losing the ball and incurring a reward of −1 (losing the game). At time point 3, the agent hits the ball by pressing 'up', and the expected reward keeps increasing until time point 4, when the ball bounces to the left side of the screen and the value of all actions reflects that the agent is about to receive a reward of 1. Note, the dashed line shows the past trajectory of the ball purely for illustrative purposes (that is, not shown during the game).

<div class="figure" style="text-align: center">
<img src="figure-qvalue.png" alt="Visualization of the learned action-value function in the game Pong" width="2598" />
<p class="caption">(\#fig:qvalue)Visualization of the learned action-value function in the game Pong</p>
</div>

## Q-network: approximator of Q-value function

Achieving optimal target values: $r + \gamma \max_{a'}Q^*(s',a')$ is hard and computationally expensive, especially when the state space is large. To address this issue, we can approximate the target values using a deep neural network as an approximator: 

$$
r + \gamma \max_{a'}Q(s',a';\theta_i^-)
$$

where $\theta_i^-$ are the parameters from some previous iteration. The Q-network are defined as a neural network approximator with weights $\theta$. The Q-network is trained to minimize the mean square error in the loss function: 

$$
L_i(\theta_i) = \mathbb{E}_{s,a,r,s'}[(\underbrace{r + \gamma \max_{a'}Q(s',a';\theta_i^-)}_{\text{target value}} - \underbrace{Q(s,a;\theta_i)}_{\text{estimation}})^2]
$$ 

and the gradient of the loss function can be written as:

$$
\nabla_{\theta_i}L_i(\theta_i) = \mathbb{E}_{s,a,r,s'}\left[\left(r + \gamma \max_{a'}Q(s',a';\theta_i^-) - Q(s,a;\theta_i)\right)\nabla_{\theta_i}Q(s,a;\theta_i)\right]
$$

The target value: $r + \gamma \max_{a'}Q(s',a';\theta_i^-)$ can be viewed as the reward you will get using parameters from some previous iteration, and the estimation: $Q(s,a;\theta_i)$ is the reward you will get using the current parameters. The loss function is the squared difference between the target value and the estimation.

# Deep Reinforcement Learning

The paper by Mnih et al. (2015) expands the Q-learning algorithm which described above by using a deep neural network to approximate the Q-value function with 3 new features covered below.

## Convolutional network {#network}

<div class="figure" style="text-align: center">
<img src="fig-network.png" alt="Visualization of the network architecture" width="720" />
<p class="caption">(\#fig:network)Visualization of the network architecture</p>
</div>

The first new features used in the paper is the convolutional network used to process the image of the games into states which will be fed into the Q-network (shown in Fig \@ref(fig:network)).

The details of what is a convolutional network is beyond the scope of this project, but in short, a convolutional network is a type of neural network that is well-suited for processing images. More can be found in the following video made by 3Blue1Brown

```{=html}
<div class="vembedr" align="center">
<div>
<iframe src="https://www.youtube.com/embed/KuXjwB4LzSA" width="533" height="300" frameborder="0" allowfullscreen="" data-external="1"></iframe>
</div>
</div>
```


Before feeding the images to the network, some pre-processing need to be done as the raw images of the emulator (Atari 2600), which are 210 × 160 pixel images with a 128-colour palette, is demanding in terms of computation and memory requirements. First, to encode a single frame the authors take the maximum value for each pixel colour value over the frame being encoded and the previous frame. This is necessary to remove flickering that is present in games where some objects appear only in even frames while other objects appear only in odd frames, an artefact caused by the limited number of sprites Atari 2600 can display at once. Second, authors then extract the Y channel, also known as luminance, from the RGB frame and rescale it to 84 × 84. The function ($\phi$) from algorithm 1 described below applies this pre-processing to the 4 most recent frames and stacks them to produce the input to the Q-function.

After image preprocessing, an 84 × 84 × 4 image will be used to train the network. The first hidden layer convolves 32 filters of 8 × 8 with stride 4 with the input image and applies a rectifier nonlinearity. The second hidden layer convolves 64 filters of 4 × 4 with stride 2, again followed by a rectifier nonlinearity. This is followed by a third convolutional layer that convolves 64 filters of 3 × 3 with stride 1 followed by a rectifier. The final hidden layer is fully-connected and consists of 512 rectifier units. The output layer is a fully-connected linear layer with a single output for each valid action (Q-value).

<div class="figure" style="text-align: center">
<img src="figure-tsne.png" alt="Two-dimensional t-SNE visualization of the representations in the last hidden layer assigned by the network to game states experienced while playing Space Invaders" width="720" />
<p class="caption">(\#fig:tsne)Two-dimensional t-SNE visualization of the representations in the last hidden layer assigned by the network to game states experienced while playing Space Invaders</p>
</div>

Fig \@ref(fig:tsne) shows that the convolutional network is able to learn the representation of the game states that are useful for predicting the Q-values of the actions. The plots are generated by running t-SNE algorithem on the last hidden layer representations assigned by the network to game states experienced while playing Space Invaders. The dots are colored according to $V$: the maximum expected reward of a state predicted by the agent (network) for the corresponding game states. The agent can separate full screens and nearly empty screens while predicting both of them with highest values because it learned that completing a screen leads to a new screen full of enemy ships. Partially completed screens are also separated from the full and empty screens, and the agent predicts them with lower values because less reward is available. The agent also learned to neglect the the orange bunkers when the game is near to the next level as they are not important for the reward.

## Experience replay

The second new feature is the experience replay. The experience replay is a technique that randomly samples previous experiences from the agent's memory and uses them to train the network. This technique is used to break the correlation between consecutive samples and stabilize the learning process. In order words, it prevent the agent from learning from only recent experiences, which are highly correlated with current state. The experience replay stores the agent's experiences in a replay memory, which is a dataset of tuples $(s, a, r, s')$. Only the current state $s$, action $a$, reward $r$, and next state $s'$ are stored as any previous information is irrelevant for the agent to make the action. The agent samples a minibatch of experiences from the replay memory and uses them to train the network. The replay memory has a fixed size and when it is full, the oldest experiences are removed to make space for new experiences.

## Second target network $\widehat{Q}$

The third new feature is the introduction of the target network $\hat{Q}$. The agent will use $\hat{Q}$ instead of $Q$ to generates targets $y_j = r + \gamma \max_{a'}Q(s',a';\theta_i^-)$ on each update. This will make divergence less likely to happen, as an update that increase $Q(s_t, a_t)$ often also increase $Q(s_{t+1},a) \forall a$. This implies that the target $y_j$ might be increasing as well, leading to oscillations or divergence. By seting the target being calculated by another network which no not update for a certain period, the target will be fixed for a while, and making the learning process more stable. The target network in this study is set to be a copy of the Q-network, but only updated periodically.

| Game           | With replay, with target Q | With replay, without target Q | Without replay, with target Q | Without replay, without target Q |
|---------------|---------------|---------------|---------------|---------------|
| Breakout       | 316.8                      | 240.7                         | 10.2                          | 3.2                              |
| Enduro         | 1006.3                     | 831.4                         | 141.9                         | 29.1                             |
| River Raid     | 7446.6                     | 4102.8                        | 2867.7                        | 1453.0                           |
| Seaquest       | 2894.4                     | 822.6                         | 1003.0                        | 275.8                            |
| Space Invaders | 1088.9                     | 826.3                         | 373.2                         | 302.0                            |

: (#tab:tab1) Comparison of the performance of the deep Q-network agent with and without experience replay and target Q-network on five Atari 2600 games. The value representing the highest average episode score. The results are taken from Mnih et al. (2015)

Table \@ref(tab:tab1) shows that the agent benefits from both experience replay and target Q-network, with the highest score achieved when both techniques are used.

## Model Training

The authors use RMSProp algorithm with a mini-batch size of 32 to train the agent. The behaviour policy during training is $\epsilon$-greedy policy with $\epsilon$ annealed linearly from 1 to 0.1 over the first million frames and fixed at 0.1 thereafter. The agent is trained for 50 million frames, which is equivalent to 38 days of game time and use a replay memory of size one million. Both the reward and errors ($r + \gamma \max_{a'}Q(s',a';\theta_i^-) - Q(s,a;\theta_i)$) are clipped at [-1,1], and actions are only selected on every fourth frame and repeated for the next three frames. This is less expensive to run and has little effect on the performance as the fastest humane player can only react every 6th frame. Hyperparameters are selected by 'informal search' (not e.g. grid search).

## Algorithm

**Deep Q-learning with experience replay**.    

| Initialize replay memory $D$ to capacity $N$;      
| Initialize action-value function $Q$ with random weights $\theta$;     
| Initialize target action-value function $\hat{Q}$ with weights $\theta^- = \theta$;     
| **FOR** episode = 1, $M$ **DO**.     
|     Initialize sequence $s_1 = \{x_1\}$ and preprocessed sequenced $\phi_1 = \phi(s_1)$;     
|     **FOR** $t = 1, T$ **DO**.    
|         With probability $\epsilon$ select a random action $a_t$;     
|         otherwise select $a_t = \max_a Q(\phi(s_t), a; \theta)$;     
|         Execute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$;     
|         Set $s_{t+1} = s_t, a_t, x_{t+1}$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$;      
|         Store transition $(\phi_t, a_t, r_t, \phi_{t+1})$ in $D$;       
|         Sample random minibatch of transitions $(\phi_j, a_j, r_j, \phi_{j+1})$ from $D$;       
|         Set $y_j = \begin{cases} r_j & \text{for terminal } \phi_{j+1} \\ r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^-) & \text{for non-terminal } \phi_{j+1} \end{cases}$;      
|         Perform a gradient descent step on $(y_j - Q(\phi_j, a_j; \theta))^2$ with respect to the network parameters $\theta$;      
|         Every $C$ steps reset $\hat{Q} = Q$;     
|     **END FOR**.    
| **END FOR**.    

The above algorithm is describe in the paper by Mnih et al. (2015) and is used to train the deep Q-network agent. episode is defined as an entire gameplay. The greedy parameter $\epsilon$ allows the agent to explore the environment by selecting a random action with probability $\epsilon$ and selecting the action with the highest Q-value with probability $1-\epsilon$. $\phi$ is the preprocessing function described above in \@ref(network). The target value $y_j$ is calculated as $r_j$ if the next state is terminal(end of the game), and $r_j + \gamma \max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^-)$ if the next state is non-terminal. The target network $\hat{Q}$ is updated every $C = 10000$ steps to be the same as the Q-network $Q$. 

## Results and evaluation

<div class="figure" style="text-align: center">
<img src="figure-traincurve.png" alt="Traning curves tracking the agent's average score and average predicted action value" width="720" />
<p class="caption">(\#fig:traincurve)Traning curves tracking the agent's average score and average predicted action value</p>
</div>

Fig \@ref(fig:traincurve) shows the agent's average score and average predicted action value over the course of training. The agent's average score increases over time as the agent learns to play the game better. The average predicted action value also increases over time, indicating that the agent is learning to predict the value of actions more accurately. In easier games, the agent learns to play the game well quickly, and the scores are more stable, while in harder games, the agent takes longer to learn to play the game well and the results are more variable.

<div class="figure" style="text-align: center">
<img src="figure-evaluation.png" alt="Comparison of the DQN agent with the best reinforcement learning methods in the literature" width="720" />
<p class="caption">(\#fig:evaluation)Comparison of the DQN agent with the best reinforcement learning methods in the literature</p>
</div>

The authors further evaluate the agent in all 49 games in the Atari 2600 platform and compare the results with the best reinforcement learning methods in the literature. Fig \@ref(fig:evaluation) shows the comparison of the deep Q-network agent with the best reinforcement learning methods in the literature. The deep Q-network agent outperforms the best reinforcement learning methods in the literature in the majority of the games, and outperform professional human game testers in more than half of the games. It is worth to notice that the best performance games are those easier to play and easier for the agent to learn (e.g: Boxing, Breakout, Star gunner, Pinball, etc.). These games' strategies can be summarized as opening a 'hole' (shown in video), which is easier for the agent to learn. And the reward in clear and immediate as well. But for games that is difficult to play, or the reward is vague and may take a long time to achieve, the agent may not perform well. Montezuma's Revenge is one of the hardest games in the Atari 2600 platform, which require the player to control the character to explore the maze and collect the keys to open the doors (showed in Fig \@ref(fig:mont)). The reward will only be given when the player reach the end of the maze and open the door. The evaluation shows that the agent is basically play the game randomly (0% performance) and did not learn the strategy to play the game.

<div class="figure" style="text-align: center">
<img src="figure-mont.png" alt="Screenshot of the game Montezuma's Revenge" width="349" />
<p class="caption">(\#fig:mont)Screenshot of the game Montezuma's Revenge</p>
</div>

This video [@youtube] demonstrates how an agent is showing improvement over training episodes and is able to pick up the optimal strategy at hitting the bricks to gain higher score in the game. 


```{=html}
<div class="vembedr" align="center">
<div>
<iframe src="https://www.youtube.com/embed/TmPfTpjtdgg" width="533" height="300" frameborder="0" allowfullscreen="" data-external="1"></iframe>
</div>
</div>
```

## Summary
-   Deep reinforcement learning is adapted from reinforcement learning, with abilities to learn more complex policies from high dimensional sensory input
-   The goal is to maximize the cumulative future reward
-   Implement convolutional neural network as an approximator for the target values 
-   Perform experience replay to remove correlations and stores the agent's experiences in a replay memory
-   Minimize the mean square error between Q-network and Q-learning target
-   The algorithm uses stochastic gradient descent to update the weights
-   The DQN was tested on 49 Atari 2600 games, and it outperforms other reinforcement learning algorithms


# References

::: {#refs}
:::
