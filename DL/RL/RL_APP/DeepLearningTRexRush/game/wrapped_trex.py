import pygame
import random
from trex_utils import *
import os

FPS = 60
scr_size = (width, height) = (600, 150)
gravity = 0.6

high_score = 0

background_col = (235, 235, 235)

pygame.init()
FPSCLOCK = pygame.time.Clock()
try:
    screen = pygame.display.set_mode(scr_size)
except pygame.error:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    screen = pygame.display.set_mode(scr_size)
pygame.display.set_caption('T-Rex Rush')


class Dino():
    def __init__(self, sizex=-1, sizey=-1):
        self.images, self.rect = load_sprite_sheet('dino.png', 5, 1, sizex, sizey, -1)
        self.images1, self.rect1 = load_sprite_sheet('dino_ducking.png', 2, 1, 59, sizey, -1)
        self.rect.bottom = int(0.98 * height)
        self.rect.left = width / 15
        self.image = self.images[0]
        self.index = 0
        self.counter = 0
        self.score = 0
        self.isJumping = False
        self.isDead = False
        self.isDucking = False
        self.isBlinking = False
        self.movement = [0, 0]
        self.jumpSpeed = 11.5

        self.stand_pos_width = self.rect.width
        self.duck_pos_width = self.rect1.width

    def draw(self):
        screen.blit(self.image, self.rect)

    def checkbounds(self):
        if self.rect.bottom > int(0.98 * height):
            self.rect.bottom = int(0.98 * height)
            self.isJumping = False

    def update(self):
        if self.isJumping:
            self.movement[1] = self.movement[1] + gravity

        if self.isJumping:
            self.index = 0
        elif self.isBlinking:
            if self.index == 0:
                if self.counter % 400 == 399:
                    self.index = (self.index + 1) % 2
            else:
                if self.counter % 20 == 19:
                    self.index = (self.index + 1) % 2

        elif self.isDucking:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2
        else:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2 + 2

        if self.isDead:
            self.index = 4

        if not self.isDucking:
            self.image = self.images[self.index]
            self.rect.width = self.stand_pos_width
        else:
            self.image = self.images1[(self.index) % 2]
            self.rect.width = self.duck_pos_width

        self.rect = self.rect.move(self.movement)
        self.checkbounds()

        if not self.isDead and self.counter % 7 == 6 and self.isBlinking == False:
            self.score += 1
            # if self.score % 100 == 0 and self.score != 0:
            # if pygame.mixer.get_init() != None:
            #    checkPoint_sound.play()

        self.counter = (self.counter + 1)


class Cactus(pygame.sprite.Sprite):
    def __init__(self, speed=5, sizex=-1, sizey=-1):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images, self.rect = load_sprite_sheet('cacti-small.png', 3, 1, sizex, sizey, -1)
        self.rect.bottom = int(0.98 * height)
        self.rect.left = width + self.rect.width
        self.image = self.images[random.randrange(0, 3)]
        self.movement = [-1 * speed, 0]

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)

        if self.rect.right < 0:
            self.kill()


class Ptera(pygame.sprite.Sprite):
    def __init__(self, speed=5, sizex=-1, sizey=-1):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images, self.rect = load_sprite_sheet('ptera.png', 2, 1, sizex, sizey, -1)
        self.ptera_height = [height * 0.82, height * 0.75, height * 0.60]
        self.rect.centery = self.ptera_height[random.randrange(0, 3)]
        self.rect.left = width + self.rect.width
        self.image = self.images[0]
        self.movement = [-1 * speed, 0]
        self.index = 0
        self.counter = 0

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        if self.counter % 10 == 0:
            self.index = (self.index + 1) % 2
        self.image = self.images[self.index]
        self.rect = self.rect.move(self.movement)
        self.counter = (self.counter + 1)
        if self.rect.right < 0:
            self.kill()


class Ground():
    def __init__(self, speed=-5):
        self.image, self.rect = load_image('ground.png', -1, -1, -1)
        self.image1, self.rect1 = load_image('ground.png', -1, -1, -1)
        self.rect.bottom = height
        self.rect1.bottom = height
        self.rect1.left = self.rect.right
        self.speed = speed

    def draw(self):
        screen.blit(self.image, self.rect)
        screen.blit(self.image1, self.rect1)

    def update(self):
        self.rect.left += self.speed
        self.rect1.left += self.speed

        if self.rect.right < 0:
            self.rect.left = self.rect1.right

        if self.rect1.right < 0:
            self.rect1.left = self.rect.right


class Cloud(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image('cloud.png', int(90 * 30 / 42), 30, -1)
        self.speed = 1
        self.rect.left = x
        self.rect.top = y
        self.movement = [-1 * self.speed, 0]

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()


class Scoreboard():
    def __init__(self, x=-1, y=-1):
        self.score = 0
        self.tempimages, self.temprect = load_sprite_sheet('numbers.png', 12, 1, 11, int(11 * 6 / 5), -1)
        self.image = pygame.Surface((55, int(11 * 6 / 5)))
        self.rect = self.image.get_rect()
        if x == -1:
            self.rect.left = width * 0.89
        else:
            self.rect.left = x
        if y == -1:
            self.rect.top = height * 0.1
        else:
            self.rect.top = y

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self, score):
        score_digits = extractDigits(score)
        self.image.fill(background_col)
        for s in score_digits:
            self.image.blit(self.tempimages[s], self.temprect)
            self.temprect.left += self.temprect.width
        self.temprect.left = 0


class GameState:
    def __init__(self):
        self.dino = Dino(44, 47)
        self.cacti = pygame.sprite.Group()
        self.clouds = pygame.sprite.Group()
        self.pteras = pygame.sprite.Group()
        self.last_obstacle = pygame.sprite.Group()

        Cactus.containers = self.cacti
        Ptera.containers = self.pteras
        Cloud.containers = self.clouds

        self.gamespeed = 4
        self.ground = Ground(-1 * self.gamespeed)
        self.counter = 0
        self.scb = Scoreboard()
        self.highsc = Scoreboard(width * 0.78)

        temp_images, temp_rect = load_sprite_sheet('numbers.png', 12, 1, 11, int(11 * 6 / 5), -1)
        self.HI_image = pygame.Surface((22, int(11 * 6 / 5)))
        self.HI_rect = self.HI_image.get_rect()
        self.HI_image.fill(background_col)
        self.HI_image.blit(temp_images[10], temp_rect)
        temp_rect.left += temp_rect.width
        self.HI_image.blit(temp_images[11], temp_rect)
        self.HI_rect.top = height * 0.1
        self.HI_rect.left = width * 0.73

    def frame_step(self, input_actions):
        global high_score
        pygame.event.pump()

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: jump
        # input_actions[2] == 1: duck # removed according to http://cs229.stanford.edu/proj2016/report/KeZhaoWei-AIForChromeOfflineDinosaurGame-report.pdf

        if input_actions[1] == 1:
            if self.dino.rect.bottom == int(0.98 * height) and not self.dino.isDucking:
                self.dino.isJumping = True
                self.dino.movement[1] = -1 * self.dino.jumpSpeed

        # if input_actions[2] == 1:
        #    if not self.dino.isJumping:
        #        self.dino.isDucking = True

        for c in self.cacti:
            c.movement[0] = -1 * self.gamespeed
            if pygame.sprite.collide_mask(self.dino, c):
                self.dino.isDead = True

        for p in self.pteras:
            p.movement[0] = -1 * self.gamespeed
            if pygame.sprite.collide_mask(self.dino, p):
                self.dino.isDead = True

        if len(self.cacti) < 2:
            if len(self.cacti) == 0:
                self.last_obstacle.empty()
                self.last_obstacle.add(Cactus(self.gamespeed, 40, 40))
            else:
                for l in self.last_obstacle:
                    if l.rect.right < width * 0.7 and random.randrange(0, 50) == 10:
                        self.last_obstacle.empty()
                        self.last_obstacle.add(Cactus(self.gamespeed, 40, 40))

        if len(self.pteras) == 0 and random.randrange(0, 200) == 10 and self.counter > 500:
            for l in self.last_obstacle:
                if l.rect.right < width * 0.8:
                    self.last_obstacle.empty()
                    self.last_obstacle.add(Ptera(self.gamespeed, 46, 40))

        if len(self.clouds) < 5 and random.randrange(0, 300) == 10:
            Cloud(width, random.randrange(height / 5, height / 2))

        self.dino.update()
        self.cacti.update()
        self.pteras.update()
        self.clouds.update()
        self.ground.update()
        self.scb.update(self.dino.score)
        self.highsc.update(high_score)

        if self.dino.isDead:
            terminal = True
            if self.dino.score > high_score:
                high_score = self.dino.score
            self.__init__()
            reward = -100
        # else:
        #    reward = self.dino.score * 0.1

        screen.fill(background_col)
        self.ground.draw()
        self.clouds.draw(screen)
        self.scb.draw()
        if high_score != 0:
            self.highsc.draw()
            screen.blit(self.HI_image, self.HI_rect)
        self.cacti.draw(screen)
        self.pteras.draw(screen)
        self.dino.draw()

        if self.counter % 700 == 699:
            self.ground.speed -= 1
            self.gamespeed += 1

        self.counter += 1

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)

        return image_data, reward, terminal
