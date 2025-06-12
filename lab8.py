import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
import torch
import time

# Verify GPU
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
if not torch.cuda.is_available():
    print("Warning: GPU not available. Training will use CPU, which is slower. Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")

# Environment definition
class RoboticArmEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, render=False, client_id=None):
        super(RoboticArmEnv, self).__init__()
        self.render_mode = 'human' if render else None
        try:
            if client_id is None:
                # Disconnect any existing connections
                try:
                    p.disconnect()
                except:
                    pass
                self.client = p.connect(p.GUI if render else p.DIRECT)
            else:
                self.client = client_id
            if self.client < 0:
                raise RuntimeError("Failed to connect to PyBullet physics server")
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            self.plane = p.loadURDF("plane.urdf")
            try:
                self.arm = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
            except:
                print("Warning: 'kuka_iiwa/model.urdf' not found. Using fallback URDF.")
                self.arm = p.loadURDF("r2d2.urdf", [0, 0, 0], useFixedBase=True)
            try:
                self.cup = p.loadURDF("cube_small.urdf", [0.5, 0, 0.1])
            except:
                print("Warning: 'cube_small.urdf' not found. Using fallback URDF.")
                self.cup = p.loadURDF("duck_vhacd.urdf", [0.5, 0, 0.1])
            self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(6,), dtype=np.float32)
            self.observation_space = spaces.Dict({
                "joint_angles": spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32),
                "cup_pos": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "image": spaces.Box(low=0, high=1, shape=(64, 64, 4), dtype=np.float32)
            })
            self.max_steps = 100
            self.step_count = 0
            self.joint_indices = list(range(min(6, p.getNumJoints(self.arm))))
        except Exception as e:
            print(f"Error initializing PyBullet: {e}")
            raise

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        for i in self.joint_indices:
            p.resetJointState(self.arm, i, 0)
        cup_x = np.random.uniform(0, 1)
        p.resetBasePositionAndOrientation(self.cup, [cup_x, 0, 0.1], [0, 0, 0, 1])
        return self._get_obs(), {}

    def step(self, action):
        current_angles = [p.getJointState(self.arm, i)[0] for i in self.joint_indices]
        new_angles = np.clip(current_angles + action[:len(self.joint_indices)], -np.pi, np.pi)
        for i, angle in enumerate(new_angles):
            p.setJointMotorControl2(self.arm, self.joint_indices[i], p.POSITION_CONTROL, targetPosition=angle)
        p.stepSimulation(clientId=self.client)
        cup_pos, cup_orient = p.getBasePositionAndOrientation(self.cup, physicsClientId=self.client)
        euler = p.getEulerFromQuaternion(cup_orient)
        end_effector_pos = self._get_end_effector_pos()
        reward = float(-0.1 * np.linalg.norm(cup_pos[0] - end_effector_pos[0]))
        terminated = False
        truncated = False
        self.step_count += 1
        if abs(euler[0]) > np.pi/2 or abs(euler[1]) > np.pi/2:
            reward += 100.0
            terminated = True
        elif abs(cup_pos[1]) > 0.1 or cup_pos[2] < 0 or cup_pos[2] > 0.2:
            reward -= 50.0
            terminated = True
        elif self.step_count >= self.max_steps:
            reward -= 100.0
            truncated = True
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        joint_angles = np.array([p.getJointState(self.arm, i)[0] for i in self.joint_indices], dtype=np.float32)
        if len(joint_angles) < 6:
            joint_angles = np.pad(joint_angles, (0, 6 - len(joint_angles)), mode='constant')
        cup_pos = np.array([p.getBasePositionAndOrientation(self.cup, physicsClientId=self.client)[0][0]], dtype=np.float32)
        image = self._capture_camera()
        return {
            "joint_angles": joint_angles,
            "cup_pos": cup_pos,
            "image": image
        }

    def _capture_camera(self):
        return np.random.rand(64, 64, 4).astype(np.float32)

    def _get_end_effector_pos(self):
        link_index = min(6, p.getNumJoints(self.arm) - 1)
        return np.array([p.getLinkState(self.arm, link_index, physicsClientId=self.client)[0][0], 0, 0.1], dtype=np.float32)

    def render(self, mode='human'):
        if mode == 'human' and self.render_mode == 'human':
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0.5, 0, 0.5], physicsClientId=self.client
            )
            time.sleep(0.05)  # Slow down for visualization

    def close(self):
        try:
            p.disconnect(self.client)
        except:
            pass

# Custom callback for plotting
class PlotCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PlotCallback, self).__init__(verbose)
        self.rewards = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if 'reward' in self.locals:
            self.rewards.append(self.locals['reward'])
        if self.locals.get('done', False):
            self.episode_rewards.append(sum(self.rewards[-self.locals['n_steps']:]))
        return True

    def _on_training_end(self) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title('Training Progress')
        plt.show()

# Create a single PyBullet client for both training and evaluation
try:
    p.disconnect()  # Ensure no existing connections
except:
    pass
shared_client = p.connect(p.DIRECT)

# Validate environment
env = RoboticArmEnv(client_id=shared_client)
check_env(env)
print("Environment is valid!")

# Train with GPU
start_time = time.time()
model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./tb_logs/",
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
model.learn(total_timesteps=100_000, callback=PlotCallback(), tb_log_name="ppo_robotic_arm")
model.save("ppo_robotic_arm")
print(f"Training time: {(time.time() - start_time)/60:.2f} minutes")

# Close training environment
env.close()

# Switch to GUI for evaluation
try:
    p.disconnect(shared_client)
except:
    pass
shared_client = p.connect(p.GUI)

# Evaluate and visualize trained arm
def evaluate_model(model, env, n_eval_episodes=5, record_video=True):  # Reduced for quicker visualization
    """Visualize the trained robotic arm in PyBullet GUI and optionally record video."""
    episode_rewards = []
    episode_lengths = []
    if record_video:
        video_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "robotic_arm.mp4", physicsClientId=env.client)
    print("Visualizing trained robotic arm...")
    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            env.render()  # Show arm movements in GUI
            done = terminated or truncated
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Steps = {steps}")
    if record_video:
        p.stopStateLogging(video_id, physicsClientId=env.client)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(episode_rewards)
    plt.xlabel('Episode Reward')
    plt.title('Reward Distribution')
    plt.subplot(1, 2, 2)
    plt.hist(episode_lengths)
    plt.xlabel('Episode Steps')
    plt.title('Length Distribution')
    plt.tight_layout()
    plt.show()
    print(f"Success rate: {sum(r > 50 for r in episode_rewards)/n_eval_episodes*100:.2f}%")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")


env = RoboticArmEnv(render=True)
p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "robotic_arm.mp4")
model = PPO.load("ppo_robotic_arm")
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
p.stopStateLogging()
env.close()