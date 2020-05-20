import pygame
import random

WIN_W = 1800
WIN_H = 1000
BALL_SIZE = 15
ball_vel = 20
PAD_H = 180
Ball_img = pygame.transform.scale2x(pygame.image.load("Pong.png"))
Ball_img = pygame.transform.scale(Ball_img, (BALL_SIZE*2,BALL_SIZE*2))
Paddle_img = pygame.transform.scale(pygame.image.load("PongPlayer.png"), (20, PAD_H))
PAD_S = 15

win = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("Pong")
win.fill((0,0,0))


def draw_env():
	pygame.draw.rect(win, (255,255,255), pygame.Rect((20, 20), (WIN_W - 40, 10)))
	pygame.draw.rect(win, (255,255,255), pygame.Rect((20, WIN_H-30), (WIN_W - 40, 10)))

class Paddle1:
	def __init__(self, y):
		self.y = y
		self.x = 20
		win.blit(Paddle_img, (self.x, self.y))

	def draw(self):
		global ball
		if ball.y >= 40 and ball.y <= WIN_H - PAD_H - 40:
			self.y = ball.y
		win.blit(Paddle_img, (self.x, self.y))

	def get_mask(self):
		return pygame.mask.from_surface(Paddle_img)


class Paddle2:
	def __init__(self, y):
		self.y = y
		self.x = 1770
		win.blit(Paddle_img, (self.x, self.y))

	def draw(self):
		win.blit(Paddle_img, (self.x, self.y))

	def get_mask(self):
		return pygame.mask.from_surface(Paddle_img)

	def move_up(self):
		if self.y >=40:
			self.y -= PAD_S

	def move_down(self):
		if self.y <= WIN_H - PAD_H - 40:
			self.y += PAD_S


class Ball:
	def __init__(self, x, y):
		self.x = x
		self.y = y

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

		pong2mask = paddle2.get_mask()
		pong1mask = paddle1.get_mask()
		if mask.overlap(pong1mask, (paddle1.x-self.x, paddle1.y-self.y)):
			self.xvel = -self.xvel


		elif mask.overlap(pong2mask, (paddle2.x-self.x, paddle2.y-self.y)):
			self.xvel = -self.xvel

		

		self.x += self.xvel
		self.y += self.yvel

		



ball = Ball(900, 500)
paddle1 = Paddle1(int((WIN_H - PAD_H)/2))
paddle2 = Paddle2(int((WIN_H - PAD_H)/2))

clock = pygame.time.Clock()

game_running = True
while game_running:
	round_active = True
	ball = Ball(900, 500)
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

		if ball.x <= 0 or ball.x >= WIN_W:
			break

		win.fill((0,0,0))
		draw_env()
		paddle1.draw()
		paddle2.draw()

		ball.move()
		ball.draw()
		pygame.display.update()
		clock.tick(60)






