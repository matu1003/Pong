import pygame
import random
from math import cos, sin, pi
pygame.font.init()

STAT_FONT = pygame.font.SysFont("arial", 200)
WIN_W = 1500
WIN_H = 1000
BALL_SIZE = 10
ball_vel_start = 20
ball_vel = ball_vel_start
PAD_H = 100
Ball_img = pygame.transform.scale2x(pygame.image.load("Pong.png"))
Ball_img = pygame.transform.scale(Ball_img, (BALL_SIZE*2,BALL_SIZE*2))
Paddle_img = pygame.transform.scale(pygame.image.load("PongPlayer.png"), (10, PAD_H))
CenterLineImg = pygame.transform.scale(pygame.image.load("CenterLine.png"), (5, WIN_H-60))
PAD_S1 = 10
PAD_S2 = 15
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
		self.next_check = 10
		self.pred_y = WIN_H/2
	def draw(self):
		win.blit(Paddle_img, (self.x, self.y))

	def get_mask(self):
		return pygame.mask.from_surface(Paddle_img)

	def move_up(self):
		if self.y >=40:
			self.y -= PAD_S1

	def move_down(self):
		if self.y <= WIN_H - PAD_H - 40:
			self.y += PAD_S1

	def move(self):
		global ball
		if ball.xvel >= 0:
			self.next_check = 1
			self.pred_y = WIN_H/2

		elif ball.yvel == 0:
			self.pred_y = ball.y
		elif self.next_check == 0:
			x_sim = ball.x
			y_sim = ball.y
			xvel = ball.xvel
			yvel = ball.yvel
			while x_sim >= 30:
				y_arr = 30
				if yvel > 0:
					y_arr = WIN_H - 30
				x_sim += ((y_arr-y_sim)/yvel) * xvel
				yvel = -yvel
				y_sim = y_arr
			self.pred_y =  abs(((30-x_sim)/xvel) * yvel)
			if y_sim == WIN_H-30:
				self.pred_y = WIN_H - self.pred_y
			self.next_check = 10
		else:
			self.next_check -= 1

		if (self.y + PAD_H/2) + PAD_S1/2 < self.pred_y:
			self.move_down()
		elif (self.y + PAD_H/2) - PAD_S1/2 > self.pred_y:
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

	def move_down(self):
		if self.y <= WIN_H - PAD_H - 40:
			self.y += PAD_S2


class Ball:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.check_delay = check_delay
		# self.xvel = -6
		# self.yvel = -30
		global ball_vel
		self.yvel = round(random.randint(0,ball_vel*100) / 200)

		direction = random.randint(0,1)
		self.xvel = round((ball_vel**2 - self.yvel**2)**0.5)
		if direction == 0:
			self.xvel = -self.xvel


	def draw(self):
		win.blit(Ball_img, (self.x, self.y))

	def move(self):
		if (self.y < 30 and self.yvel < 0) or (self.y > WIN_H - 30 - 2*BALL_SIZE and self.yvel > 0):
			self.yvel = -self.yvel

		mask = pygame.mask.from_surface(Ball_img)
		global paddle1
		global paddle2

		pong2mask = paddle2.get_mask()
		pong1mask = paddle1.get_mask()

		alpha = 0
		global strokes
		if self.check_delay == 0:

			if mask.overlap(pong1mask, (paddle1.x-self.x, paddle1.y-self.y)):
				print(ball_vel)
				strokes += 1
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

			elif mask.overlap(pong2mask, (paddle2.x-self.x, paddle2.y-self.y)):
				print(ball_vel)
				strokes += 1
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

		else:
			self.check_delay -= 1
		

		self.x += self.xvel
		self.y += self.yvel

		



ball = Ball(int(WIN_W/2), int(WIN_H/2))
paddle1 = Paddle1(int((WIN_H - PAD_H)/2))
paddle2 = Paddle2(int((WIN_H - PAD_H)/2))
scores = [0, 0]

clock = pygame.time.Clock()

game_running = True
while game_running:
	ball_vel = ball_vel_start
	strokes = 0
	round_active = True
	ball = Ball(750, 500)
	paddle1 = Paddle1(int((WIN_H - PAD_H)/2))
	paddle2 = Paddle2(int((WIN_H - PAD_H)/2))

	clock = pygame.time.Clock()
	while round_active:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				game_running = False
				round_active = False
				break
		
		keys=pygame.key.get_pressed()
		if keys[pygame.K_DOWN]:
			paddle2.move_down()
		if keys[pygame.K_UP]:
			paddle2.move_up()

		if ball.x <= -20:
			scores[1] += 1
			break

		elif ball.x >= WIN_W + 20:
			scores[0] += 1
			break
		if strokes == 10 and ball_vel <= 32:
			ball_vel += 1
			strokes = 0

		win.fill((0,0,0))

		draw_env()
		score1 = STAT_FONT.render(str(scores[0]), 1, (255, 255, 255))
		win.blit(score1, (int(WIN_W/4), 60))
		score2 = STAT_FONT.render(str(scores[1]), 2, (255, 255, 255))
		win.blit(score2, (int(2*WIN_W/3), 60))
		paddle1.move()
		paddle1.draw()
		paddle2.draw()

		ball.move()
		ball.draw()
		pygame.display.update()
		clock.tick(60)






