import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

class Environment:
    def __init__(self, variants, payouts, n_trials):
        self.variants = variants
        self.payouts = payouts
        self.n_trials = n_trials
        self.total_reward = 0
        self.n_k = len(variants)
        self.shape = (self.n_k, n_trials)
        
    def run(self, agent):
        """Run the simulation with the agent. 
        agent must be a class with choose_k and update methods."""
        
        for i in range(self.n_trials):
            # agent makes a choice
            x_chosen = agent.choose_k()
            # Environment returns reward
            reward = np.random.binomial(1, p=self.payouts[x_chosen])
            # agent learns of reward
            agent.reward = reward
            # agent updates parameters based on the data
            agent.update()
            self.total_reward += reward
        
        agent.collect_data()
        
        return self.total_reward


machines = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
payouts = [0.023, 0.001, 0.029, 0.001, 0.002, 0.04, 0.0234, 0.002, 0.01, 0.0121, .3]

class BaseSampler:
    
    def __init__(self, env, n_samples=None, n_learning=None, e=0.05):
        self.env = env
        self.shape = (env.n_k, n_samples)
        self.variants = env.variants
        self.n_trials = env.n_trials
        self.payouts = env.payouts
        self.ad_i = np.zeros(env.n_trials)
        self.r_i = np.zeros(env.n_trials)
        self.regret_i = np.zeros(env.n_trials)
        self.beta_post = np.random.uniform(0, 1, size=self.shape)
        
        self.a = np.ones(env.n_k) 
        self.b = np.ones(env.n_k) 
        self.theta = np.zeros(env.n_k)
        self.data = None
        self.reward = 0
        self.total_reward = 0
        self.k = 0
        self.i = 0
        
        
        self.n_samples = n_samples
        self.n_learning = n_learning
        self.e = e
        self.ep = np.random.uniform(0, 1, size=env.n_trials)
        self.exploit = (1 - e)

class eGreedy(BaseSampler):

    def __init__(self, env, n_learning, e):
        super().__init__(env, n_learning, e)
        
    def choose_k(self):

        # e% of the time take a random draw from machines
        # random k for n learning trials, then the machine with highest theta
        self.k = np.random.choice(self.variants) if self.i < self.n_learning else np.argmax(self.theta)
        # with 1 - e probability take a random sample (explore) otherwise exploit
        self.k = np.random.choice(self.variants) if self.ep[self.i] > self.exploit else self.k
        return self.k
        # every 100 trials update the successes

        # update the count of successes for the chosen machine
    def update(self):
        
        self.a[self.k] += self.reward
        self.b[self.k] += 1
        # update the probability of payout for each machine
        self.theta = self.a/self.b

        self.total_reward += self.reward
        self.regret_i[self.i] = np.max(self.theta) - self.theta[self.k]
        self.ad_i[self.i] = self.k
        self.r_i[self.i] = self.reward
        self.i += 1
    
    def collect_data(self):
        
        self.data = pd.DataFrame(dict(ad=self.ad_i, reward=self.r_i, regret=self.regret_i))

class ThompsonSampler(BaseSampler):

    def __init__(self, env, n_samples):
        super().__init__(env, n_samples)
        
    def choose_k(self):

        self.beta_post[self.k, :] = np.random.beta(self.a[self.k], self.b[self.k], size=self.shape)[self.k]

        for self.k in range(self.env.n_k):
            # sample from posterior (this is the thompson sampling approach)
            # this leads to more exploration because machines with > uncertainty can then be selected as the machine
            #xpost[k, :] = xpost[k, :][np.round(self.beta_post[k, :], 3) != 0]
            self.theta[self.k] = np.random.choice(self.beta_post[self.k, :])
        
        # select machine with highest posterior p of payout
        self.k = self.variants[np.argmax(self.theta)]
        return self.k
    
    def update(self):
        
        self.regret_i[self.i] = np.max(self.beta_post) - self.theta[self.k]
        #update dist (a, b) = (a, b) + (r, 1 - r) 
        self.a[self.k] += self.reward
        self.b[self.k] += 1 - self.reward # i.e. only increment b when it's a swing and a miss. 1 - 0 = 1, 1 - 1 = 0
        self.total_reward += self.reward
        self.ad_i[self.i] = self.k
        self.r_i[self.i] = self.reward
        self.i += 1
    
    def collect_data(self):
        self.data = pd.DataFrame(dict(ad=self.ad_i, reward=self.r_i, regret=self.regret_i))

en = Environment(machines, payouts, 10000)
tsa = ThompsonSampler(env=en, n_samples=50)
en.run(agent=tsa)

en = Environment(machines, payouts, 10000)
tsa = eGreedy(env=en, n_learning=1000, e=0.05)
en.run(agent=tsa)

