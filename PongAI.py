import pygame
import random
from math import cos, sin, pi
import tensorflow as tf
from tensorflow import keras
import numpy as np


pygame.font.init()

STAT_FONT = pygame.font.SysFont("arial", 200)
WIN_W = 1500
WIN_H = 1000
BALL_SIZE = 8
ball_vel_start = 25
ball_vel = ball_vel_start
PAD_H = 75
Ball_img = pygame.transform.scale2x(pygame.image.load("Pong.png"))
Ball_img = pygame.transform.scale(Ball_img, (BALL_SIZE*2,BALL_SIZE*2))
Paddle_img = pygame.transform.scale(pygame.image.load("PongPlayer.png"), (10, PAD_H))
CenterLineImg = pygame.transform.scale(pygame.image.load("CenterLine.png"), (5, WIN_H-60))
PAD_S1 = 20
PAD_S2 = 20
check_delay = 2

win = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("Pong")
win.fill((0,0,0))


def draw_env():
	pygame.draw.rect(win, (255,255,255), pygame.Rect((20, 20), (WIN_W - 40, 10)))
	pygame.draw.rect(win, (255,255,255), pygame.Rect((20, WIN_H-30), (WIN_W - 40, 10)))
	win.blit(CenterLineImg, (int(WIN_W/2), 30))

class Paddle1:
	def __init__(self, y):
		self.y = y
		self.x = 20
		win.blit(Paddle_img, (self.x, self.y))

	def draw(self):
		win.blit(Paddle_img, (self.x, self.y))

	def get_mask(self):
		return pygame.mask.from_surface(Paddle_img)

	def move_up(self):
		if self.y >=40:
			self.y -= PAD_S1

	def move_down(self):
		if self.y <= WIN_H - PAD_H - 20:
			self.y += PAD_S1

	def move(self):
		global ball_vel
		if self.y < ball.y:
			self.move_down()
		elif self.y + PAD_H > ball.y:
			self.move_up()


class Paddle2:
	def __init__(self, y):
		self.y = y
		self.x = WIN_W -30
		win.blit(Paddle_img, (self.x, self.y))

	def draw(self):
		win.blit(Paddle_img, (self.x, self.y))

	def get_mask(self):
		return pygame.mask.from_surface(Paddle_img)

	def move_up(self):
		if self.y >=40:
			self.y -= PAD_S2
			return True
		else:
			return False

	def move_down(self):
		if self.y <= WIN_H - PAD_H - 40:
			self.y += PAD_S2
			return True
		else:
			return False


class Ball:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.check_delay = check_delay

		global ball_vel
		self.yvel = round(random.randint(0,ball_vel*100) / 200)

		direction = random.randint(0,1)
		self.xvel = round((ball_vel**2 - self.yvel**2)**0.5)
		if direction == 0:
			self.xvel = -self.xvel


	def draw(self):
		win.blit(Ball_img, (self.x, self.y))

	def move(self):
		if self.y < 30 or self.y > WIN_H - 30 - 2*BALL_SIZE:
			self.yvel = -self.yvel

		mask = pygame.mask.from_surface(Ball_img)
		global paddle1
		global paddle2
		global strokes

		pong2mask = paddle2.get_mask()
		pong1mask = paddle1.get_mask()

		alpha = 0
		impact = 0

		if self.check_delay == 0:

			if mask.overlap(pong1mask, (paddle1.x-self.x, paddle1.y-self.y)):
				section = PAD_H / 7
				alphas = [-45, -30, -15, 0, 15, 30, 45]
				for sect in range(7):
					if self.y <= paddle1.y + (sect+1) * section:
						alpha = alphas[sect]*2*pi/360
						break

				#Simple
				# self.xvel = -self.xvel
				#Real
				self.xvel = int(cos(alpha) * ball_vel)
				self.yvel = int(sin(alpha) * ball_vel)
				self.check_delay = check_delay
				strokes += 1

			elif mask.overlap(pong2mask, (paddle2.x-self.x, paddle2.y-self.y)):

				section = PAD_H / 7
				alphas = [-45, -30, -15, 0, 15, 30, 45]
				for sect in range(7):
					if self.y <= paddle2.y + (sect+1) * section:
						alpha = alphas[sect]*2*pi/360
						break

				#Simple
				# self.xvel = -self.xvel
				#Real
				self.xvel = int(-cos(alpha) * ball_vel)
				self.yvel = int(sin(alpha) * ball_vel)

				self.check_delay = check_delay
				impact = 1
				strokes += 1

		else:
			self.check_delay -= 1
		

		self.x += self.xvel
		self.y += self.yvel
		return impact

		



ball = None
paddle1 = None
paddle2 = None
scores = None
done = None
win.fill((0,0,0))
pygame.display.update()
clock = pygame.time.Clock()
scores = [0,0]
strokes = 0

def reset():
	global ball, paddle1, paddle2, scores, done, win, strokes
	ball = Ball(int(WIN_W/2), int(WIN_H/2))
	paddle1 = Paddle1(int((WIN_H - PAD_H)/2))
	paddle2 = Paddle2(int((WIN_H - PAD_H)/2))
	done = False
	win.fill((0,0,0))
	draw_env()
	paddle1.draw()
	ball.draw()
	paddle2.draw()
	pygame.display.update()
	obs = np.array([ball.y/1000, ball.x/1000, paddle2.y/1000, ball.yvel/1000, ball.xvel/1000])
	strokes = 0
	return obs


def step(action):
	global done, ball, paddle1, paddle2, ball_vel
	ball_vel = ball_vel_start + (strokes //10)
	win.fill((0,0,0))
	draw_env()
	if not(done):
		act = True
		if action == 0:
			act = paddle2.move_down()
		elif action == 1:
			act = paddle2.move_up()
		paddle1.move()
		paddle1.draw()
		paddle2.draw()
		imp = ball.move()
		ball.draw()


	score1 = STAT_FONT.render(str(scores[0]), 1, (255, 255, 255))
	win.blit(score1, (int(WIN_W/4), 60))
	score2 = STAT_FONT.render(str(scores[1]), 2, (255, 255, 255))
	win.blit(score2, (int(2*WIN_W/3), 60))
	pygame.display.update()
	
	obs = np.array([ball.y/1000, ball.x/1000, paddle2.y/1000, ball.yvel/1000, ball.xvel/1000])
	reward = 0


	if ball.x <= 0:
		scores[1] += 1
		reward = 0.5
		done = True

	elif ball.x >= WIN_W:
		scores[0] += 1
		reward = -5
		done = True
		reward += 5 - (5*abs(paddle2.y+(PAD_H/2)-ball.y)/WIN_H)
		print(reward)

	if imp == 1:
		reward += 10

	if act == False:
		reward -= 0.5
	return obs, float(reward), done


reset()


n_inputs = 5

model = keras.models.Sequential([
	keras.layers.Dense(3, activation="sigmoid", input_shape = [n_inputs]),
	keras.layers.Dense(10, activation="sigmoid"),
	keras.layers.Dense(1, activation="sigmoid")
])


def play_one_step(obs, model, loss_fn):
	with tf.GradientTape() as tape:
		left_proba = model(obs[np.newaxis])
		action = (tf.random.uniform([1,1]) > left_proba)
		y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
		loss = tf.reduce_mean(loss_fn(y_target, left_proba))
	grads = tape.gradient(loss, model.trainable_variables)
	obs, reward, done = step(int(action[0,0].numpy()))
	return obs, reward, done, grads


def play_multiple_episodes(n_episodes, n_max_steps, model, loss_fn):
	all_rewards = []
	all_grads = []
	for episode in range(n_episodes):
		current_rewards = []
		current_grads = []
		obs = reset()
		for step in range(n_max_steps):
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pass
			obs, reward, done, grads = play_one_step(obs, model, loss_fn)
			current_rewards.append(reward)
			current_grads.append(grads)
			if done:
				break
		all_rewards.append(current_rewards)
		all_grads.append(current_grads)
	return all_rewards, all_grads

def discount_rewards(rewards, discount_factor):
	discounted = np.array(rewards)
	for step in range(len(rewards)-2, -1, -1):
		discounted[step] += discounted[step+1] * discount_factor
	return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
	all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]
	flat_rewards = np.concatenate(all_discounted_rewards)
	reward_mean = flat_rewards.mean()
	print(reward_mean)
	reward_std = flat_rewards.std()
	return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]	

n_iterations = 15000
n_episodes_per_update = 10
n_max_steps = 1000000000000000
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=0.02)
loss_fn = keras.losses.binary_crossentropy

for iteration in range(n_iterations):
	all_rewards, all_grads = play_multiple_episodes(n_episodes_per_update, n_max_steps, model, loss_fn)
	all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)
	all_mean_grads = []
	for var_index in range(len(model.trainable_variables)):
		mean_grads = tf.reduce_mean(
		[final_reward * all_grads[episode_index][step][var_index] 
		for episode_index, final_rewards in enumerate(all_final_rewards) for step, final_reward in enumerate(final_rewards)], axis = 0)
		all_mean_grads.append(mean_grads)
	optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
	print(iteration)




'''
winopen = True
while winopen:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			winopen = False
			break
	obs, reward, done = step(int(input("Action: ")))
	print(obs)
	if done:
		reset()
'''







